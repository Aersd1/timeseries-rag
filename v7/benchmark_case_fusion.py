"""Train identical adapters with/without additional five/ten cases; freeze policy before TEST."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .common import sha256,write_json
from .aligned_case_fusion import MODES,CaseFusion,inputs,tensor_inputs,forecast,align_cases
from .prepare_case_fusion import prepare
from .benchmark_state_recall import metric,interval
from .safe_error_policy import bound_delta

SETTINGS=dict(width=32,batch_size=256,epochs=48,patience=10,learning_rate=.001,
    weight_decay=.001,seeds=[13,41,73],prediction_patch=8,alignment_shrink=5.,
    training_loss='TRAIN-scale weighted MSE + 0.05 MAE + 0.001 correction energy',
    validation='First two per-variable VAL blocks select epoch/beta; last confirms/rejects only')


def load_trace(folder,split,expected=None):
    path=folder/f'{split}.npz'
    if expected is not None and sha256(path)!=expected:raise ValueError('Changed feature/label cache')
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def parts(data):
    calibration=[];confirmation=[]
    for sid in np.unique(data['sid']):
        rows=np.flatnonzero(data['sid']==sid);rows=rows[np.argsort(data['start'][rows])]
        blocks=np.array_split(rows,3);calibration.extend(np.r_[blocks[0],blocks[1]].tolist());confirmation.extend(blocks[2].tolist())
    return np.asarray(calibration,dtype=int),np.asarray(confirmation,dtype=int)


def train_model(train,val,stats,period,mode,seed,folder,device):
    torch.manual_seed(seed);np.random.seed(seed)
    if device.startswith('cuda'):torch.cuda.manual_seed_all(seed)
    conditioned=bool(stats.get('residual_reference'));paired=bool(stats.get('paired_attention'));metadata_width=int(stats.get('metadata_width',5))
    model=CaseFusion(len(stats['std']),SETTINGS['width'],conditioned,paired,metadata_width).to(device)
    features,_,scale=inputs(train,stats,period,mode);features=tensor_inputs(features,device)
    vfeatures,_,vscale=inputs(val,stats,period,mode);vfeatures=tensor_inputs(vfeatures,device)
    cal,_=parts(val);cal_data={k:val[k][cal] for k in ['raw_quantiles','raw_y','sid']}
    if conditioned:cal_data['reference']=val['reference'][cal]
    cal_features={k:v[cal] for k,v in vfeatures.items()}
    train_base=train['reference'] if conditioned else train['raw_quantiles'][:,4]
    val_base=val['reference'] if conditioned else val['raw_quantiles'][:,4]
    target=torch.as_tensor((train['raw_y']-train_base)/scale[:,None],dtype=torch.float32,device=device)
    weight=torch.as_tensor(np.clip((scale/np.asarray(stats['std'])[train['sid']])**2,.01,100),dtype=torch.float32,device=device)
    teacher=None
    if stats.get('predictive_contrastive'):
        level=train['raw_x'].mean(1);true_future=np.clip((train['raw_y']-level[:,None])/scale[:,None],-8,8)
        reference=np.clip((train_base-level[:,None])/scale[:,None],-8,8)
        possible=torch.cat([features['future'],torch.as_tensor(reference,dtype=torch.float32,device=device)[:,None]],1)
        errors=(possible-torch.as_tensor(true_future,dtype=torch.float32,device=device)[:,None]).square().mean(-1)
        teacher=torch.softmax(-errors/errors.median(1,keepdim=True).values.clamp_min(.1),dim=-1).detach()
    def score(pred):return float(np.mean(((pred.astype(float)-val['raw_y'][cal])/np.asarray(stats['std'])[val['sid'][cal],None])**2))
    best=score(val_base[cal]);best_epoch=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=SETTINGS['learning_rate'],weight_decay=SETTINGS['weight_decay'])
    rng=np.random.default_rng(seed);trace=[];stale=0
    for epoch in range(1,SETTINGS['epochs']+1):
        model.train();order=rng.permutation(len(scale));loss_sum=0.
        for begin in range(0,len(order),SETTINGS['batch_size']):
            ids=torch.as_tensor(order[begin:begin+SETTINGS['batch_size']],device=device)
            if teacher is None:correction=model(**{k:v[ids] for k,v in features.items()});scores=None
            else:correction,scores=model(**{k:v[ids] for k,v in features.items()},return_scores=True)
            error=correction-target[ids]
            loss=((error.square()+.05*error.abs()).mean(1)*weight[ids]).mean()+.001*correction.square().mean()
            if teacher is not None:loss=loss+.05*torch.nn.functional.kl_div(torch.log_softmax(scores,dim=-1),teacher[ids],reduction='batchmean')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();loss_sum+=float(loss.detach())*len(ids)
        pred=forecast(model,cal_features,cal_data,vscale[cal]);value=score(pred)
        trace.append(dict(epoch=epoch,training_loss=loss_sum/len(scale),calibration_loss=value))
        if value<best*(1-1e-5):
            best=value;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
        else:stale+=1
        if epoch%8==0:print('FUSION TRAIN',folder.name,mode,seed,epoch,'best',best_epoch,best,flush=True)
        if stale>=SETTINGS['patience']:break
    model.load_state_dict(best_state);model.eval()
    name=f'{mode}_{seed}.pt';torch.save(dict(state_dict=best_state,variables=len(stats['std']),width=SETTINGS['width'],conditioned=conditioned,paired=paired,metadata_width=metadata_width),folder/name)
    predictions=forecast(model,vfeatures,val,vscale)
    del features,vfeatures,target,weight,optimizer
    return model,predictions,dict(mode=mode,seed=seed,best_epoch=best_epoch,best_loss=best,trace=trace,sha256=sha256(folder/name),file=name)


def calibrate(val,pred,base=None):
    # The legacy reason labels are retained for frozen-policy replay; with
    # residual-reference inputs, rejection preserves F5 rather than Moirai.
    policies={};base=val.get('reference',val['raw_quantiles'][:,4]) if base is None else np.asarray(base);y=val['raw_y']
    for sid in np.unique(val['sid']):
        rows=np.flatnonzero(val['sid']==sid);rows=rows[np.argsort(val['start'][rows])];blocks=np.array_split(rows,3)
        default=dict(beta=0.,reason='no stable improvement over frozen Moirai')
        if min(map(len,blocks))<3:policies[str(int(sid))]=default;continue
        options=[]
        for beta in [.1,.25,.5,1.]:
            proposed=base+beta*(pred-base);gains=[]
            for ids in blocks[:2]:
                old=y[ids]-base[ids];new=y[ids]-proposed[ids];gain=np.mean(old**2,1)-np.mean(new**2,1)
                if gain.mean()<=.01*np.mean(old**2) or np.mean(gain>0)<.75 or np.mean(abs(new))>=np.mean(abs(old)):break
                gains.extend(gain.tolist())
            else:options.append((-float(np.mean(gains)),beta))
        if not options:policies[str(int(sid))]=default;continue
        _,beta=min(options);ids=blocks[2];proposed=base+beta*(pred-base)
        old=y[ids]-base[ids];new=y[ids]-proposed[ids];gain=np.mean(old**2,1)-np.mean(new**2,1)
        ok=gain.mean()>0 and np.mean(gain>0)>=.75 and np.mean(abs(new))<np.mean(abs(old))
        policies[str(int(sid))]=dict(beta=beta if ok else 0.,proposed_beta=beta,final_gain=float(gain.mean()),
            reason='confirmed' if ok else 'confirmation rejected; frozen Moirai')
    return policies


def apply(policy,data,pred,base=None):
    beta=np.array([policy[str(int(s))]['beta'] for s in data['sid']],dtype=np.float32)
    base=data.get('reference',data['raw_quantiles'][:,4]) if base is None else np.asarray(base)
    return base+beta[:,None]*(pred-base)


def mean_predictions(models,data,stats,period,mode,device):
    features,_,scale=inputs(data,stats,period,mode);features=tensor_inputs(features,device)
    individual=[forecast(model,features,data,scale) for model in models]
    return np.mean(individual,axis=0).astype(np.float32),individual


def fixed_alignment(data,stats):
    base=data['raw_quantiles'][:,4];out={}
    for mode in ['legacy','wiener']:
        _,cases=align_cases(data['raw_x'],data['case_past'],data['case_future'],mode,np.asarray(stats['memory_std'])[data['sid']])
        out['alignment_'+mode]=base+.25*bound_delta(cases.mean(1)-base,data['raw_quantiles'])
    return out


def experiment(folder,device):
    if (folder/'run.json').exists():raise ValueError('Do not overwrite a frozen completed experiment')
    preparation=json.loads((folder/'preparation.json').read_text());source=Path(preparation['source'])
    stats=json.loads((folder/'stats.json').read_text());period=preparation['period']
    if sha256(folder/'stats.json')!=preparation['stats_sha256']:raise ValueError('Changed TRAIN statistics')
    catalog=json.loads((Path(preparation['store'])/'catalog.json').read_text());stats['memory_std']=[max(s['memory_std'],1e-6) for s in catalog['series']]
    train=load_trace(folder,'train',preparation['files']['train']['sha256']);val=load_trace(folder,'validation',preparation['files']['validation']['sha256'])
    models={};val_pred={};individual_validation={};selection=[]
    for mode in MODES:
        models[mode]=[];individual_validation[mode]=[]
        for seed in SETTINGS['seeds']:
            model,pred,record=train_model(train,val,stats,period,mode,seed,folder,device)
            models[mode].append(model);individual_validation[mode].append(pred);selection.append(record)
        val_pred[mode]=np.mean(individual_validation[mode],axis=0).astype(np.float32)
    policies={mode:calibrate(val,val_pred[mode]) for mode in MODES}
    if preparation.get('case_count')==5 and not preparation.get('residual_reference'):
        with np.load(source/'validation_trace.npz',allow_pickle=False) as f:
            for key in ['sid','start']:np.testing.assert_array_equal(f[key],val[key])
            incumbent=f['incumbent'].copy()
        for mode in MODES:
            prediction=incumbent+val_pred[mode]-val['raw_quantiles'][:,4]
            policies['increment_'+mode]=calibrate(val,prediction,incumbent)
        difference=incumbent+val_pred['wiener']-val_pred['without_retrieval']
        policies['increment_difference']=calibrate(val,difference,incumbent)
    write_json(folder/'selection.json',dict(settings=SETTINGS,models=selection))
    write_json(folder/'policy.json',policies)
    np.savez_compressed(folder/'validation_forecasts.npz',**val_pred,
        **{f'{mode}_{seed}':pred for mode,values in individual_validation.items() for seed,pred in zip(SETTINGS['seeds'],values)})
    write_json(folder/'contract.json',dict(preparation_sha256=sha256(folder/'preparation.json'),stats_sha256=sha256(folder/'stats.json'),
        selection_sha256=sha256(folder/'selection.json'),policy_sha256=sha256(folder/'policy.json'),
        validation_predictions_sha256=sha256(folder/'validation_forecasts.npz'),source_run_sha256=sha256(source/'run.json'),
        primary='Learn residual of actual five-case reference' if preparation.get('residual_reference') else ('Five-case increment_wiener against previous five forecast' if preparation.get('case_count')==5 else 'wiener ensemble of fixed three seeds; no per-dataset test-best selection'),
        mechanism='Frozen Moirai2 + small trainable patch-conditioned five/ten-case attention; same architecture query-only control',
        caveat='Previously explored periods and pre-existing validation-selected retriever; frozen retriever is TRAIN-trained, not forward OOF within TRAIN'))
    # The new models, epoch choices and validation-only policies are now frozen.
    del train
    if preparation.get('residual_reference'):
        from .prepare_case_reference import prepare as prepare_reference,copy_feature
        test_info=copy_feature(Path(preparation['reference_feature_source']),folder,'test') if preparation.get('reference_feature_source') else prepare_reference(Path(preparation['reference_source']),folder,'test')
    else:test_info=prepare(source,Path(preparation['store']).parent,folder,'test')
    test=load_trace(folder,'test',test_info['sha256']);forecasts=dict(moirai_direct=test['raw_quantiles'][:,4]);test_individual={}
    for mode in MODES:
        raw,individual=mean_predictions(models[mode],test,stats,period,mode,device)
        forecasts['raw_'+mode]=raw;forecasts['calibrated_'+mode]=apply(policies[mode],test,raw)
        for seed,pred in zip(SETTINGS['seeds'],individual):test_individual[f'{mode}_{seed}']=pred
    forecasts.update(fixed_alignment(test,stats))
    with np.load(source/'predictions.npz',allow_pickle=False) as f:
        for key in ['sid','start','raw_y','std']:np.testing.assert_array_equal(test[key],f[key])
        for key in ['previous_five','guarded_dual','fixed_increment_transport','no_rag']:forecasts['previous_'+key]=f[key].copy()
    if preparation.get('case_count')==5 and not preparation.get('residual_reference'):
        incumbent=forecasts['previous_previous_five']
        for mode in MODES:
            proposal=incumbent+forecasts['raw_'+mode]-forecasts['moirai_direct']
            forecasts['raw_increment_'+mode]=proposal
            forecasts['increment_'+mode]=apply(policies['increment_'+mode],test,proposal,incumbent)
        proposal=incumbent+forecasts['raw_wiener']-forecasts['raw_without_retrieval']
        forecasts['raw_increment_difference']=proposal
        forecasts['increment_difference']=apply(policies['increment_difference'],test,proposal,incumbent)
    # Frozen-model future-shuffle control: only old case futures change.
    rng=np.random.default_rng(20261014);shuffled=dict(test)
    order=np.stack([rng.permutation(test['case_future'].shape[1]) for _ in test['sid']])
    shuffled['case_future']=test['case_future'][np.arange(len(order))[:,None],order]
    shuffled_pred,_=mean_predictions(models['wiener'],shuffled,stats,period,'wiener',device)
    forecasts['shuffled_wiener']=apply(policies['wiener'],test,shuffled_pred)
    if preparation.get('residual_reference'):forecasts['current_five_reference']=test['reference'];forecasts['current_no_case_reference']=test['no_reference']
    truth={k:test[k] for k in ['sid','start','raw_y','std']}
    np.savez_compressed(folder/'predictions.npz',**forecasts,**truth,selected_starts=test['selected_starts'])
    np.savez_compressed(folder/'seed_predictions.npz',**test_individual)
    metrics={k:metric(v,truth) for k,v in forecasts.items()}
    record=dict(complete=True,dataset=folder.name,queries=len(test['sid']),train_queries=preparation['files']['train']['queries'],
        variables=len(stats['std']),metrics=metrics,individual_seed_metrics={k:metric(v,truth) for k,v in test_individual.items()},
        case_count=preparation.get('case_count',10),
        active_variables={mode:[int(s) for s,p in values.items() if p['beta']>0] for mode,values in policies.items()},
        contract_sha256=sha256(folder/'contract.json'),predictions_sha256=sha256(folder/'predictions.npz'),
        seed_predictions_sha256=sha256(folder/'seed_predictions.npz'),test_info=test_info,
        vs_moirai_interval=interval(forecasts['moirai_direct'],forecasts['calibrated_wiener'],truth),
        rag_interval=interval(forecasts['calibrated_without_retrieval'],forecasts['calibrated_wiener'],truth),
        caveat='Same previously viewed sampled test windows; source retriever frozen; external adapter training only; no new Moirai fine-tuning')
    write_json(folder/'run.json',record);print('FUSION RESULT',folder.name,metrics,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--datasets',nargs='+',required=True)
    parser.add_argument('--device',default='cuda');args=parser.parse_args();torch.set_num_threads(4)
    root=Path(args.root);write_json(root/'protocol.json',dict(settings=SETTINGS,modes=MODES,datasets=args.datasets,
        prediction_length=96,history_length=96,source_candidates=100,auxiliary_cases='Read from each frozen preparation: 5 or 10',
        code_sha256={name:sha256('v7/'+name) for name in ['aligned_case_fusion.py','prepare_case_fusion.py','benchmark_case_fusion.py']}))
    for name in args.datasets:experiment(root/name,args.device)
