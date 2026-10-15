"""Post-expert chronological utility training and direct-Moirai fallback.

Primary arm is fixed as `real`. All other arms are diagnostics, never dataset-
specific test winners. Previously explored TEST and upstream reused VAL remain
explicit limitations; this is not a new independent benchmark.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .common import sha256,write_json
from .prepare_unified_residual import read
from .benchmark_unified_residual import trace
from .benchmark_unified_derangement import derange
from .selective_rag import (ARMS,SETTINGS,rows,chronological_split,options,
    selector_features,utility_targets,fit_selector,score_selector,route,
    error_metrics,nonworse,calibrate_router,diagnostic,UtilityNet)


def load_npz(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def frozen_predictions(source,split):
    if split=='validation':
        p=load_npz(source/'validation_predictions.npz');d=load_npz(source/'derangement_validation.npz')
        return {k:p[k] for k in ARMS[:5]}|{'deranged':d['raw']}
    p=load_npz(source/'predictions.npz');d=load_npz(source/'derangement_predictions.npz')
    return {k:p['raw_'+k] for k in ARMS[:5]}|{'deranged':d['raw']}


def arm_data(data,arm):return derange(data) if arm in ['deranged','case_deranged_utility'] else data


def verified_source(source):
    main=read(source/'run.json');contract=read(source/'contract.json');dr=read(source/'derangement_run.json');dc=read(source/'derangement_contract.json')
    pairs=[('contract.json',main['contract_sha256']),('predictions.npz',main['predictions_sha256']),
        ('validation_predictions.npz',contract['validation_sha256']),('stats.json',contract['stats_sha256']),
        ('derangement_contract.json',dr['contract_sha256']),('derangement_validation.npz',dc['validation_sha256']),
        ('derangement_predictions.npz',dr['predictions_sha256'])]
    for file,expected in pairs:
        if sha256(source/file)!=expected:raise ValueError('Changed upstream '+file)
    return {file:expected for file,expected in pairs}


def fit_static(data,candidates,stats,cal,confirm):
    # Uniformly average all options: fixed five/ten case candidate or one frozen expert.
    b=data['raw_quantiles'][:,4];c=candidates.mean(2).reshape(-1,96);bc=error_metrics(rows(data,cal),b[cal],stats);best=None
    for beta in SETTINGS['strengths']:
        pred=b+beta*(c-b);m=error_metrics(rows(data,cal),pred[cal],stats)
        if nonworse(m,bc) and m['mse']<bc['mse']*(1-1e-5):
            score=m['mse']/max(bc['mse'],1e-20)+m['train_nmse']/max(bc['train_nmse'],1e-20)
            if best is None or (score,beta)<best[:2]:best=(score,beta)
    if best is None:return dict(enabled=False,strength=0.)
    beta=best[1];pred=b+beta*(c-b);bm=error_metrics(rows(data,confirm),b[confirm],stats);cm=error_metrics(rows(data,confirm),pred[confirm],stats)
    return dict(enabled=bool(nonworse(cm,bm) and cm['mse']<bm['mse']*(1-1e-5)),strength=beta,confirmation_base=bm,confirmation=cm)


def experiment(source,out,device):
    if (out/'run.json').exists():raise ValueError('Completed experiment is immutable')
    if (out/'contract.json').exists():raise ValueError('Resume via verifier; do not refit a frozen policy')
    out.mkdir(parents=True,exist_ok=True);source_hashes=verified_source(source)
    meta=read(source/'preparation.json');stats=read(source/'stats.json');catalog=read(Path(meta['store'])/'catalog.json')
    train_end=max(s['train_end'] for s in catalog['series']);test_start=min(s['validation_end'] for s in catalog['series'])
    val=trace(source,'validation');vp=frozen_predictions(source,'validation');fit,cal,confirm=chronological_split(val,train_end,test_start)
    record=dict(dataset=source.name,source=str(source.resolve()),settings=SETTINGS,source_hashes=source_hashes,
        source_data_sha256=sha256(source/'validation.npz'),stats_sha256=sha256(source/'stats.json'),train_end=train_end,test_start=test_start,
        split={k:dict(rows=len(ids),starts=np.unique(val['start'][ids]).tolist(),last_label_end=int(val['start'][ids].max()+192)) for k,ids in [('fit',fit),('calibration',cal),('confirmation',confirm)]},
        temporal_contract='All selector-fit labels finish before calibration history; all calibration labels finish before confirmation history; all confirmation labels finish before TEST history',
        caveats=['Only selector fitting is chronological; frozen experts are TRAIN-fitted','Upstream retriever/choices used old VAL; not independent full-pipeline validation','Previously viewed TEST; exploratory results','Case utility chooses within the original 5+5; no candidate count change','Historical competence uses only completed VAL predictions; no TEST target updates'])
    policies={};saved=[];val_outputs={}
    for arm in ARMS:
        d=arm_data(val,arm);c=options(d,stats,meta['period'],arm,vp.get(arm))
        x=selector_features(d,c,stats,meta['period'],arm,d,c);g,h=utility_targets(d,c,stats)
        models,normal,training=fit_selector(x[fit],g[fit],h[fit],d['sid'][fit],stats,device)
        score=score_selector(models,normal,x,device)
        policies[arm]=calibrate_router(d,c,score,stats,cal,confirm);policies[arm]['static']=fit_static(d,c,stats,cal,confirm)
        path=out/f'{arm}.pt';torch.save(dict(normal=normal,models=[{k:v.cpu() for k,v in model.state_dict().items()} for model in models]),path)
        saved.append(dict(arm=arm,file=path.name,sha256=sha256(path),training=training,normal=normal))
        val_outputs[arm+'_scores']=score
        print('SELECTOR FROZEN',source.name,arm,policies[arm]['enabled'],policies[arm]['threshold'],policies[arm]['strength'],flush=True)
        del models,x,c,score,g,h
    write_json(out/'policy.json',policies);write_json(out/'selection.json',record|dict(models=saved))
    np.savez_compressed(out/'validation_scores.npz',**val_outputs)
    write_json(out/'contract.json',dict(primary='real',policy_sha256=sha256(out/'policy.json'),selection_sha256=sha256(out/'selection.json'),
        validation_scores_sha256=sha256(out/'validation_scores.npz'),source=record['source'],
        code_sha256={f:sha256('v7/'+f) for f in ['selective_rag.py','benchmark_selective_rag.py']},settings=SETTINGS))
    # No TEST labels have been loaded above. Every arm, model and policy is now fixed.
    test=trace(source,'test');tp=frozen_predictions(source,'test');base=test['raw_quantiles'][:,4];outputs={'moirai_direct':base};scores_out={};results={}
    previous=load_npz(source/'predictions.npz');outputs['previous_primary']=previous['extra_real'];outputs['old_fixed_5plus5']=previous['previous_fixed_increment_transport']
    for arm in ARMS:
        d=arm_data(test,arm);hd=arm_data(val,arm);c=options(d,stats,meta['period'],arm,tp.get(arm));hc=options(hd,stats,meta['period'],arm,vp.get(arm))
        x=selector_features(d,c,stats,meta['period'],arm,hd,hc)
        saved_model=torch.load(out/f'{arm}.pt',map_location='cpu',weights_only=True);normal=saved_model['normal'];models=[]
        for state in saved_model['models']:
            m=UtilityNet(normal['input_width'],SETTINGS['width']);m.load_state_dict(state);models.append(m.to(device).eval())
        score=score_selector(models,normal,x,device);policy=policies[arm]
        pred,mask,choice=route(base,c,score,policy['threshold'],policy['strength'],policy['enabled'])
        static=policy['static'];sp=base+static['strength']*(c.mean(2).reshape(-1,96)-base) if static['enabled'] else base.copy()
        outputs[arm]=pred;outputs[arm+'_static']=sp;scores_out[arm+'_scores']=score;scores_out[arm+'_mask']=mask;scores_out[arm+'_choice']=choice
        results[arm]=dict(metrics=error_metrics(d,pred,stats),static_metrics=error_metrics(d,sp,stats),
            diagnostics=diagnostic(d,c,score,pred,mask,choice,stats,strength=policy['strength'] if policy['enabled'] else 0.))
        print('SELECTOR TEST',source.name,arm,results[arm]['metrics']['mse'],'coverage',float(mask.mean()),flush=True)
        del x,c,hc,models,score
    for k in ['sid','start','raw_y','std']:outputs[k]=test[k]
    np.savez_compressed(out/'predictions.npz',**outputs);np.savez_compressed(out/'routing.npz',**scores_out)
    result=dict(complete=True,dataset=source.name,queries=len(base),variables=len(stats['std']),primary='real',
        baselines={k:error_metrics(test,outputs[k],stats) for k in ['moirai_direct','previous_primary','old_fixed_5plus5']},results=results,
        contract_sha256=sha256(out/'contract.json'),predictions_sha256=sha256(out/'predictions.npz'),routing_sha256=sha256(out/'routing.npz'))
    write_json(out/'run.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4)
    for name in a.datasets:experiment(Path(a.source)/name,Path(a.output)/name,a.device)
