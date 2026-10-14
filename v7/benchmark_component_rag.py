"""Paired cached evaluation: fixed encoders, case detail and horizon policies."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import write_json,sha256
from .query_future_state_rag import FutureStatePredictor
from .benchmark_error_correction import load
from .benchmark_state_recall import metric,interval
from .component_rag import component_experts,choose_segments,apply_segments
from .safe_error_policy import choose_policy,apply_policy


def collect(predictor,split):
    data=load(predictor.cache,split)
    probes=Path(predictor.base_run['source_probes']);manifest=probes/'manifest.json'
    if sha256(manifest)!=predictor.base_run['source_probe_manifest_sha256']:raise ValueError('Changed probe manifest')
    meta=json.loads(manifest.read_text(encoding='utf-8'))
    if sha256(probes/f'{split}.npz')!=meta['files'][split]:raise ValueError('Changed probe predictions')
    with np.load(probes/f'{split}.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(f['sid'],data['sid']);np.testing.assert_array_equal(f['start'],data['start'])
        probe=f['quantiles'].copy()
    records=[]
    for i in range(len(data['sid'])):
        sid=int(data['sid'][i]);start=int(data['start'][i])
        meta={k:data[k][i] for k in ['weights','selected_starts','candidate_starts','history_scores','future_scores']}
        result=predictor.predict_cached(data['raw_x'][i],sid,start,data['raw_quantiles'][i:i+1],probe[i:i+1],meta,True,split=='validation')
        experts,confidence,effective=component_experts(result['components'],result['weights'],result['selected_starts'])
        records.append(dict(base=result['components']['base'],final_base=result['components']['final_base'],
            disabled=result['components']['disabled'],confidence=confidence,effective=effective,
            **{f'expert_{k}':v for k,v in experts.items()},**result['forecasts']))
        if (i+1)%250==0:print('COMPONENT TRACE',predictor.folder.name,split,i+1,'/',len(data['sid']),flush=True)
    trace={k:np.stack([r[k] for r in records]) for k in records[0]}
    trace.update({k:data[k] for k in ['raw_y','std','sid','start']})
    return trace


def fit_policies(val,old):
    experts={k[7:]:v for k,v in val.items() if k.startswith('expert_')}
    target=(val['raw_y']-val['base'])/val['std'][:,None]
    extended=choose_policy(target,experts,val['sid'],val['start'],require_mae=True)
    for sid in np.unique(val['sid']):
        if np.any(val['disabled'][val['sid']==sid]):extended[str(int(sid))].update(family='direct',beta=0.)
    policies=dict(detail_global=extended,
        segmented_rag=choose_segments(target,experts,val['sid'],val['start'],old['routed'],val['disabled']),
        segmented_no_rag=choose_segments(target,{'encoder_only':experts['encoder_only']},val['sid'],val['start'],old['encoder_only'],val['disabled']))
    return policies


def forecasts(trace,policies):
    experts={k[7:]:v for k,v in trace.items() if k.startswith('expert_')}
    predictions={k:trace[k] for k in ['moirai_direct','no_rag','encoder_only','routed']}
    predictions['detail_global']=trace['final_base']+apply_policy(policies['detail_global'],experts,trace['sid'])*trace['std'][:,None]
    predictions['segmented_rag']=trace['final_base']+apply_segments(policies['segmented_rag'],experts,trace['sid'])*trace['std'][:,None]
    predictions['segmented_no_rag']=trace['final_base']+apply_segments(policies['segmented_no_rag'],{'encoder_only':experts['encoder_only']},trace['sid'])*trace['std'][:,None]
    return predictions


def experiment(checkpoint,source,out,resume=False):
    if resume and (out/'run.json').exists():raise ValueError('Completed experiment cannot be resumed')
    out.mkdir(parents=True,exist_ok=resume)
    predictor=FutureStatePredictor(source/'store',checkpoint,adapter=False)
    try:
        trace_source=dict(source_run_sha256=sha256(checkpoint/'run.json'),cache_manifest_sha256=sha256(predictor.cache/'manifest.json'),
            probe_manifest_sha256=sha256(Path(predictor.base_run['source_probes'])/'manifest.json'))
        if resume and (out/'validation_trace.npz').exists():
            if not (out/'trace_source.json').exists() or json.loads((out/'trace_source.json').read_text(encoding='utf-8'))!=trace_source:
                raise ValueError('Cannot resume a trace without matching frozen source provenance')
        else:write_json(out/'trace_source.json',trace_source)
        if resume and (out/'validation_trace.npz').exists():
            with np.load(out/'validation_trace.npz',allow_pickle=False) as f:val={k:f[k].copy() for k in f.files}
        else:val=collect(predictor,'validation');np.savez_compressed(out/'validation_trace.npz',**val)
        policies=fit_policies(val,predictor.policies);write_json(out/'policy.json',policies)
        # Freeze before collecting or inspecting test outcomes. All variants reported.
        write_json(out/'contract.json',dict(source_run_sha256=sha256(checkpoint/'run.json'),
            policy_sha256=sha256(out/'policy.json'),validation_sha256=sha256(out/'validation_trace.npz'),
            component_width=16,shrinkage=.5,recall=100,selected=5,prior=[.8,.2],
            caveat='Reused development test periods; not a new blind test'))
        test=collect(predictor,'test');np.savez_compressed(out/'test_trace.npz',**test)
        with np.load(checkpoint/'predictions.npz',allow_pickle=False) as original:
            for kind in ['moirai_direct','no_rag','encoder_only','routed']:
                np.testing.assert_allclose(test[kind],original[kind],atol=2e-4,rtol=2e-5,err_msg=f'Baseline replay {kind}')
        pred=forecasts(test,policies);truth={k:test[k] for k in ['raw_y','std','sid','start']}
        np.savez_compressed(out/'predictions.npz',**pred,**truth)
        metrics={k:metric(v,truth) for k,v in pred.items()}
        result=dict(complete=True,dataset=checkpoint.name,queries=len(test['sid']),metrics=metrics,
            accepted_segments={k:sum(p['accepted'] for p in policies[k].values()) for k in ['segmented_rag','segmented_no_rag']},
            policy_sha256=sha256(out/'policy.json'),contract_sha256=sha256(out/'contract.json'),
            source_checkpoint=str(checkpoint),predictions_sha256=sha256(out/'predictions.npz'),
            rag_gain_interval=interval(pred['segmented_no_rag'],pred['segmented_rag'],truth),
            caveat='Exploration on reused periods; no post-test choice of winning variant')
        write_json(out/'run.json',result);print('COMPONENT RESULT',checkpoint.name,metrics,result['accepted_segments'],flush=True)
    finally:predictor.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--output',required=True);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    for name in a.datasets:experiment(Path(a.checkpoint_root)/name,Path(a.source)/name,Path(a.output)/name,a.resume)
