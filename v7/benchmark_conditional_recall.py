"""Factorial recall/memory comparison without changing the learned encoder."""
import argparse
import json
from pathlib import Path
import numpy as np
from .conditional_recall import ConditionalPredictor
from .benchmark_analogue_transport import policies_for,forecasts
from .benchmark_error_correction import load
from .benchmark_state_recall import metric,interval
from .common import sha256,write_json
from .analogue_transport import experts
from .safe_error_policy import choose_policy


def collect(p,split):
    data=load(p.cache,split);probes=Path(p.base_run['source_probes']);manifest=probes/'manifest.json'
    if sha256(manifest)!=p.base_run['source_probe_manifest_sha256']:raise ValueError('Changed probe manifest')
    meta=json.loads(manifest.read_text(encoding='utf-8'))
    if sha256(probes/f'{split}.npz')!=meta['files'][split]:raise ValueError('Changed probe predictions')
    with np.load(probes/f'{split}.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(f['sid'],data['sid']);np.testing.assert_array_equal(f['start'],data['start']);probe=f['quantiles'].copy()
    records=[]
    for i in range(len(data['sid'])):
        result=p.predict_cached(data['raw_x'][i],int(data['sid'][i]),int(data['start'][i]),data['raw_quantiles'][i:i+1],probe[i:i+1],return_components=True,validation=split=='validation')
        c=result['components'];values=experts(c,result['weights'])
        records.append(dict(base=c['base'],final_base=c['final_base'],disabled=c['disabled'],selected_starts=result['selected_starts'],
            **result['forecasts'],**{f'expert_{k}':v for k,v in values.items()}))
        if (i+1)%500==0:print('CONDITIONAL TRACE',p.folder.name,p.scope,split,i+1,'/',len(data['sid']),flush=True)
    table={k:np.stack([r[k] for r in records]) for k in records[0]};table.update({k:data[k] for k in ['raw_y','std','sid','start']})
    return table


def choose(val):
    policies=policies_for(val);values={k[7:]:v for k,v in val.items() if k.startswith('expert_')}
    target=(val['raw_y']-val['base'])/val['std'][:,None]
    for name,allowed in [('conditional_copied',['future_state_rag']),('conditional_routed',['encoder_only','future_state_rag','mix_0.25','mix_0.5','mix_0.75'])]:
        policies[name]=choose_policy(target,{k:values[k] for k in allowed},val['sid'],val['start'],require_mae=True)
        for sid in np.unique(val['sid']):
            if np.any(val['disabled'][val['sid']==sid]):policies[name][str(int(sid))].update(family='direct',beta=0.)
    return policies


def experiment(checkpoint,source,out,scope,recall_mode='conditional'):
    out.mkdir(parents=True,exist_ok=False);p=ConditionalPredictor(source/'store',checkpoint,scope,adapter=False,recall_mode=recall_mode)
    try:
        val=collect(p,'validation');np.savez_compressed(out/'validation_trace.npz',**val)
        policies=choose(val);write_json(out/'policy.json',policies)
        write_json(out/'contract.json',dict(source_run_sha256=sha256(checkpoint/'run.json'),memory_scope=scope,recall_mode=recall_mode,
            limits={str(s['sid']):p.memory_limit(s) for s in p.store.series},source_data_id=p.store.meta['data_id'],
            policy_sha256=sha256(out/'policy.json'),validation_sha256=sha256(out/'validation_trace.npz'),
            recall=100,selected=5,rule='Every old192-point episode ends before query history; train scope never includes validation or test outcomes',
            caveat='Frozen train memory changes information available relative to prefix-only baseline; explicit separate comparison'))
        test=collect(p,'test');np.savez_compressed(out/'test_trace.npz',**test)
        pred=forecasts(test,policies)
        # The old routed arm must retain its own original retrieval, rather than relabel altered cases.
        with np.load(checkpoint/'predictions.npz',allow_pickle=False) as original:
            np.testing.assert_array_equal(test['sid'],original['sid']);np.testing.assert_array_equal(test['start'],original['start'])
            for name in ['moirai_direct','no_rag','encoder_only']:np.testing.assert_allclose(pred[name],original[name],rtol=2e-5,atol=2e-4)
            pred['routed']=original['routed'].copy()
        truth={k:test[k] for k in ['raw_y','std','sid','start']}
        np.savez_compressed(out/'predictions.npz',**pred,**truth,selected_starts=test['selected_starts'])
        metrics={k:metric(v,truth) for k,v in pred.items()}
        result=dict(complete=True,dataset=checkpoint.name,memory_scope=scope,recall_mode=recall_mode,queries=len(test['sid']),metrics=metrics,source_checkpoint=str(checkpoint),
            policy_sha256=sha256(out/'policy.json'),contract_sha256=sha256(out/'contract.json'),predictions_sha256=sha256(out/'predictions.npz'),
            gain_interval=interval(pred['routed'],pred['conditional_routed'],truth),rag_interval=interval(pred['encoder_only'],pred['conditional_routed'],truth),
            caveat='Reused development periods; scope train is a frozen TRAIN-only bank, not online test feedback')
        write_json(out/'run.json',result);print('CONDITIONAL RESULT',checkpoint.name,scope,metrics,flush=True)
    finally:p.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint-root',required=True);parser.add_argument('--source',default='v7/runs/all_datasets_20261008')
    parser.add_argument('--output',required=True);parser.add_argument('--datasets',nargs='+',required=True);parser.add_argument('--scope',choices=['prefix','train'],required=True)
    parser.add_argument('--recall-mode',choices=['conditional','history','hybrid'],default='conditional');a=parser.parse_args()
    for name in a.datasets:experiment(Path(a.checkpoint_root)/name,Path(a.source)/name,Path(a.output)/name,a.scope,a.recall_mode)
