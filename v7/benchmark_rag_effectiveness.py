"""Mechanism experiment: frozen cached Moirai, fixed 5+5 fusion, no new backbone."""
import argparse,gc,time
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import sha256,write_json
from .benchmark_memory_access import read
from .current_forecast import DATASETS,RUN_ROOT
from .rag_effectiveness import (ALPHA,FEATURES,QUERY_FEATURES,ARMS,transfer,
    fit_sign_model,sign_probability,improvement_identity)

SOURCE=Path('v7/runs/unified_residual_5plus5_96_20261015')
CONTEXT=Path('v7/runs/history_length_full_20261016')
OPERATORS=('residual','continuation')


def load(name,split):
    source=SOURCE/name;previous=CONTEXT/name
    meta=read(source/'preparation.json');record=read(previous/f'{split}_cohort.json')
    if sha256(source/'stats.json')!=meta['stats_sha256']:raise ValueError('Changed TRAIN scale')
    if sha256(source/f'{split}.npz')!=record['source_sha256']:raise ValueError('Changed source cases')
    if sha256(previous/f'{split}_cohort.npz')!=record['cohort_sha256']:raise ValueError('Changed context cohort')
    policy=read(previous/'policy.json')
    if sha256(previous/'policy.json')!=read(CONTEXT/'frozen.json')[name]:raise ValueError('Changed context policy')
    length=policy['primary_length']
    with np.load(previous/f'{split}_cohort.npz',allow_pickle=False) as f:c={k:f[k].copy() for k in f.files}
    keys=['raw_x','raw_y','raw_quantiles','probe_quantiles','sid','start','selected_starts',
        'case_past','case_future','case_baseline','random_selected_starts',
        'random_case_past','random_case_future','random_case_baseline']
    with np.load(source/f'{split}.npz',allow_pickle=False) as f:d={k:f[k][c['source_rows']].copy() for k in keys}
    for k in ('sid','start','raw_y'):np.testing.assert_array_equal(d[k],c[k])
    np.testing.assert_array_equal(d['raw_quantiles'][:,4],c['moirai_96'])
    if length==96:base=c['moirai_96']
    else:
        manifest=read(previous/f'{split}_{length}.json');path=previous/f'{split}_{length}.npz'
        if sha256(path)!=manifest['prediction_sha256'] or manifest['cohort_sha256']!=record['cohort_sha256']:raise ValueError('Changed long baseline')
        with np.load(path,allow_pickle=False) as f:
            for k in ('sid','start'):np.testing.assert_array_equal(f[k],d[k])
            base=f['prediction'].copy()
    d['base_long']=base;d['base_short']=c['moirai_96'];d['train_std']=c['train_std']
    if split=='test':d['original5plus5']=c['restored_5plus5']
    else:
        old=RUN_ROOT/name;contract=read(old/'contract.json')
        if sha256(old/'validation_trace.npz')!=contract['source_validation_sha256']:raise ValueError('Changed original validation')
        with np.load(old/'validation_trace.npz',allow_pickle=False) as f:
            for k in ('sid','start','raw_y','selected_starts'):np.testing.assert_array_equal(f[k][c['source_rows']],d[k])
            incumbent=f['incumbent'][c['source_rows']]
            d['original5plus5']=(incumbent+.25*(f['increment_transport'][c['source_rows']]-incumbent)).copy()
    store=Store(meta['store']);physical=0
    try:
        if sha256(Path(meta['store'])/'catalog.json')!=record['catalog_sha256']:raise ValueError('Changed catalog')
        for sid in np.unique(d['sid']):
            sid=int(sid);series=store.series[sid];rows=np.flatnonzero(d['sid']==sid);raw=store.values(sid)
            if sha256(Path(meta['store'])/series['file'])!=record['values_sha256'][str(sid)]:raise ValueError('Changed raw data')
            np.testing.assert_array_equal(raw[d['start'][rows,None]+np.arange(96)],d['raw_x'][rows])
            np.testing.assert_array_equal(raw[d['start'][rows,None]+96+np.arange(96)],d['raw_y'][rows])
            for prefix in ('','random_'):
                starts=d[prefix+'selected_starts'][rows]
                if np.any(starts+192>np.minimum(d['start'][rows],meta['limits'][str(sid)])[:,None]):raise ValueError('Incomplete or overlapping case')
                if any(len(np.unique(s))!=10 for s in starts):raise ValueError('Need exact distinct5+5')
                np.testing.assert_array_equal(raw[starts[:,:,None]+np.arange(96)],d[prefix+'case_past'][rows])
                np.testing.assert_array_equal(raw[starts[:,:,None]+96+np.arange(96)],d[prefix+'case_future'][rows])
                physical+=len(rows)*10
    finally:store.close()
    evidence=dict(dataset=name,split=split,queries=len(d['sid']),origins=len(np.unique(d['start'])),
        length=length,period=record['period'],physical_case_pairs=physical,
        source_sha256=record['source_sha256'],context_cohort_sha256=record['cohort_sha256'],
        context_policy_sha256=sha256(previous/'policy.json'),preparation_sha256=sha256(source/'preparation.json'))
    return d,evidence


def compute(d,evidence,batch_size=256):
    features=[];predictions={'original5plus5':d['original5plus5']};identity_max=0.
    for baseline,length in [('short',96),('long',evidence['length'])]:
        base=d['base_'+baseline];predictions[f'{baseline}_no_rag']=base
        for arm in ARMS:
            prefix='random_' if arm=='matched_random' else ''
            parts={k:[] for k in OPERATORS};fs=[]
            for begin in range(0,len(base),batch_size):
                rows=slice(begin,begin+batch_size)
                values=transfer(d['raw_x'][rows],d['raw_quantiles'][rows],d['probe_quantiles'][rows],
                    d['sid'][rows],d['start'][rows],d[prefix+'case_past'][rows],
                    d[prefix+'case_future'][rows],d[prefix+'case_baseline'][rows],
                    d[prefix+'selected_starts'][rows],d['train_std'][rows],base[rows],length,
                    arm=arm,with_features=arm=='real' and baseline=='long')
                for operator in OPERATORS:parts[operator].append(values[operator])
                if 'features' in values:fs.append(values['features'])
            for operator in OPERATORS:predictions[f'{baseline}_{operator}_{arm}']=np.concatenate(parts[operator])
            if fs:features=np.concatenate(fs)
    losses={};truth=d['raw_y'].astype(float)
    for key,pred in predictions.items():
        error=pred.astype(float)-truth
        losses[key]=np.stack(
            [(error**2).mean(1),abs(error).mean(1),((error/d['train_std'][:,None])**2).mean(1)],1)
    for baseline in ('short','long'):
        a,e,g=improvement_identity(truth,predictions[f'{baseline}_no_rag'],predictions[f'{baseline}_residual_real'])
        actual=losses[f'{baseline}_no_rag'][:,0]-losses[f'{baseline}_residual_real'][:,0]
        identity_max=max(identity_max,float(np.max(abs(actual-(a-e)))))
    return features,predictions,losses,identity_max


def interval_losses(base,new,starts,period):
    week=(starts+96)//max(1,int(round(7*period)));sums=np.array([[base[week==w].sum(),new[week==w].sum()] for w in np.unique(week)])
    rng=np.random.default_rng(20261019);draw=sums[rng.integers(len(sums),size=(4000,len(sums)))].sum(1)
    gains=100*(1-draw[:,1]/np.maximum(draw[:,0],1e-30))
    return dict(gain_percent=float(100*(1-new.sum()/max(base.sum(),1e-30))),
        exploratory_interval95=np.quantile(gains,[.025,.975]).tolist(),weekly_clusters=len(sums))


def score_losses(losses,rows=None):
    if rows is None:rows=np.ones(len(next(iter(losses.values()))),dtype=bool)
    return {key:dict(zip(('mse','mae','train_nmse'),value[rows].mean(0).tolist())) for key,value in losses.items()}


def summarize(d,evidence,features,losses,policy,models):
    from sklearn.metrics import roc_auc_score,balanced_accuracy_score,accuracy_score
    base=losses['long_no_rag'];real=losses['long_residual_real'];gain=base[:,2]-real[:,2]
    label=gain>0;gates={};sign_scores={}
    for family,model in models.items():
        probability=sign_probability(model,features);gate=probability>=.5;gates[family]=gate
        sign_scores[family]=dict(auc=float(roc_auc_score(label,probability)) if len(np.unique(label))==2 else None,
            balanced_accuracy=float(balanced_accuracy_score(label,gate)),accuracy=float(accuracy_score(label,gate)),
            coverage=float(gate.mean()))
    gates['observable_rule']=(features[:,0]<=policy['medians'][0])&(features[:,1]<=policy['medians'][1])&(features[:,4]>0)&(features[:,5]>=.5)
    gates['val_dataset_rule']=np.full(len(base),policy['val_real_gain_nmse']>0)
    # An outcome-based oracle is an explicitly impossible upper bound.
    gates['oracle_future_only']=gain>0
    gate_scores={}
    for key,gate in gates.items():
        chosen=np.where(gate[:,None],real,base)
        gate_scores[key]=dict(coverage=float(gate.mean()),metrics=score_losses({'gated':chosen})['gated'],
            gain_vs_no_rag=interval_losses(base[:,0],chosen[:,0],d['start'],evidence['period']))
    groups={}
    for j,name in enumerate(FEATURES):
        values={}
        for label_group,mask in [('low',features[:,j]<=policy['quartiles'][j][0]),('high',features[:,j]>=policy['quartiles'][j][1])]:
            if mask.sum():
                b=base[mask].sum(0);r=real[mask].sum(0)
                values[label_group]=dict(queries=int(mask.sum()),origins=len(np.unique(d['start'][mask])),
                    gain_mse_percent=float(100*(1-r[0]/max(b[0],1e-30))),
                    gain_nmse_percent=float(100*(1-r[2]/max(b[2],1e-30))),win_rate=float((gain[mask]>0).mean()),
                    original_gain_vs96_nmse_percent=float(100*(1-losses['original5plus5'][mask,2].sum()/max(losses['short_no_rag'][mask,2].sum(),1e-30))))
        groups[name]=values
    cells={}
    for similar in (False,True):
        for consistent in (False,True):
            mask=((features[:,0]<=policy['medians'][0])==similar)&((features[:,1]<=policy['medians'][1])==consistent)
            if mask.sum():cells[f'similar_{similar}_consistent_{consistent}']=dict(queries=int(mask.sum()),
                gain_nmse_percent=float(100*(1-real[mask,2].sum()/max(base[mask,2].sum(),1e-30))))
    comparisons={}
    for baseline in ('short','long'):
        b=losses[f'{baseline}_no_rag']
        for operator in OPERATORS:
            r=losses[f'{baseline}_{operator}_real']
            rnd=losses[f'{baseline}_{operator}_matched_random']
            # Average LOSSES of four interventions, never the loss of their ensemble.
            deranged=np.mean([losses[f'{baseline}_{operator}_deranged_{j}'] for j in (1,2,3,4)],0)
            comparisons[f'{baseline}_{operator}']=dict(
                real_vs_base=interval_losses(b[:,0],r[:,0],d['start'],evidence['period']),
                real_vs_random=interval_losses(rnd[:,0],r[:,0],d['start'],evidence['period']),
                real_vs_deranged=interval_losses(deranged[:,0],r[:,0],d['start'],evidence['period']),
                real_gain_nmse_percent=float(100*(1-r[:,2].sum()/max(b[:,2].sum(),1e-30))),
                win_fraction=float((r[:,0]<b[:,0]).mean()))
    blocks=[]
    for starts in np.array_split(np.unique(d['start']),4):
        ids=np.isin(d['start'],starts);b=base[ids,0].sum();r=real[ids,0].sum()
        blocks.append(dict(queries=int(ids.sum()),gain_mse_percent=float(100*(1-r/max(b,1e-30)))))
    return dict(dataset=evidence['dataset'],queries=evidence['queries'],origins=evidence['origins'],
        baseline_length=evidence['length'],metrics=score_losses(losses),comparisons=comparisons,
        sign_prediction=sign_scores,majority_accuracy=max(float(label.mean()),float(1-label.mean())),
        gates=gate_scores,factor_groups=groups,similarity_consistency_cells=cells,blocks=blocks,
        caveats=['Explored sampled TEST, not a fresh blind benchmark',
            'Associations in real data are not causal identification; synthetic interventions use a toy forecaster',
            'Frozen long query baseline, but historical case references remain Moirai96; secondary continuation instrument checks reference dependence',
            'One inherited matched-random realization and four group-preserving residual/value derangements',
            'LODO sign coefficients exclude held-dataset labels; baseline length was already selected on its own VAL',
            'Oracle uses query outcomes and is not deployable; probability gate does not guarantee lower MSE'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output)
    root.mkdir(parents=True,exist_ok=False);begin=time.perf_counter()
    protocol=dict(datasets=list(DATASETS),alpha=ALPHA,operators=list(OPERATORS),primary='long_residual_real',
        arms=list(ARMS),features=list(FEATURES),new_forecast_network=False,query_horizon=96,
        baseline='Previous all-VAL frozen context choice; same rows as history_length_full_20261016',
        main_tests=['Real versus no RAG','Real versus state-matched random','Real versus four true within-group pair derangements',
            'Short versus long baseline interaction','VAL-cut feature subgroups on TEST','LODO VAL-fitted observable sign prediction',
            'Paired synthetic repeatability x stale association x base skill interventions'],
        gate='Fixed0.5 probability; fixed similarity/consistency median and positive old-case check rule; diagnostic only',
        no_test_selection=True,default_unchanged=True,interval='Exploratory week-cluster bootstrap,4000 draws; no multiplicity correction')
    write_json(root/'protocol.json',protocol);validation={}
    for name in DATASETS:
        d,evidence=load(name,'validation');f,preds,losses,error=compute(d,evidence)
        folder=root/name;folder.mkdir()
        label=losses['long_residual_real'][:,2]<losses['long_no_rag'][:,2]
        policy=dict(quartiles=np.quantile(f,[.25,.75],axis=0).T.tolist(),medians=np.median(f,axis=0).tolist(),
            val_real_gain_nmse=float(np.mean(losses['long_no_rag'][:,2]-losses['long_residual_real'][:,2])),
            val_evidence=evidence,identity_max_absolute_error=error)
        write_json(folder/'policy.json',policy)
        np.savez_compressed(folder/'validation_features.npz',features=f,label=label,sid=d['sid'],start=d['start'])
        validation[name]=(f,label);del d,preds,losses;gc.collect()
        print('EFFECT VAL',name,len(f),round(time.perf_counter()-begin,1),flush=True)
    for name in DATASETS:
        others=[n for n in DATASETS if n!=name]
        x=np.concatenate([validation[n][0] for n in others]);y=np.concatenate([validation[n][1] for n in others])
        weights=np.concatenate([np.full(len(validation[n][1]),1/len(validation[n][1])) for n in others])
        models={family:fit_sign_model(x,y,weights,columns) for family,columns in
            [('query_only',QUERY_FEATURES),('case_aware',tuple(range(len(FEATURES))))]}
        write_json(root/name/'sign_models.json',models)
    frozen={name:{f:sha256(root/name/f) for f in ('policy.json','sign_models.json','validation_features.npz')} for name in DATASETS}
    write_json(root/'frozen.json',frozen)
    from .synthetic_rag_effectiveness import run as synthetic
    write_json(root/'synthetic.json',synthetic())
    results=[]
    for name in DATASETS:
        folder=root/name
        for f,h in frozen[name].items():
            if sha256(folder/f)!=h:raise ValueError('Changed frozen selection')
        d,evidence=load(name,'test');features,preds,losses,error=compute(d,evidence)
        result=summarize(d,evidence,features,losses,read(folder/'policy.json'),read(folder/'sign_models.json'))
        np.savez_compressed(folder/'test_rows.npz',features=features,sid=d['sid'],start=d['start'],
            base_short=d['base_short'],base_long=d['base_long'],
            real_long=preds['long_residual_real'],real_short=preds['short_residual_real'],
            **{'loss_'+k:v for k,v in losses.items()})
        write_json(folder/'test_evidence.json',dict(evidence,identity_max_absolute_error=error,
            rows_sha256=sha256(folder/'test_rows.npz')))
        write_json(folder/'result.json',result);results.append(result)
        print('EFFECT RESULT',name,result['comparisons']['long_residual'],result['sign_prediction'],flush=True)
        del d,preds,losses;gc.collect()
    write_json(root/'summary.json',dict(complete=True,seconds=time.perf_counter()-begin,records=results,protocol=protocol))


if __name__=='__main__':main()
