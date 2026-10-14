"""Fixed-encoder comparison of copied futures and conditional analogue transport."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import sha256,write_json
from .query_future_state_rag import FutureStatePredictor
from .benchmark_error_correction import load
from .benchmark_state_recall import metric,interval
from .safe_error_policy import choose_policy,apply_policy
from .analogue_transport import experts


def collect(predictor,split):
    data=load(predictor.cache,split);probes=Path(predictor.base_run['source_probes']);manifest=probes/'manifest.json'
    if sha256(manifest)!=predictor.base_run['source_probe_manifest_sha256']:raise ValueError('Changed probe manifest')
    meta=json.loads(manifest.read_text(encoding='utf-8'))
    if sha256(probes/f'{split}.npz')!=meta['files'][split]:raise ValueError('Changed probe cache')
    with np.load(probes/f'{split}.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(f['sid'],data['sid']);np.testing.assert_array_equal(f['start'],data['start']);probe=f['quantiles'].copy()
    records=[]
    for i in range(len(data['sid'])):
        m={k:data[k][i] for k in ['weights','selected_starts','candidate_starts','history_scores','future_scores']}
        result=predictor.predict_cached(data['raw_x'][i],int(data['sid'][i]),int(data['start'][i]),data['raw_quantiles'][i:i+1],probe[i:i+1],m,True,split=='validation')
        c=result['components'];values=experts(c,result['weights'])
        records.append(dict(base=c['base'],final_base=c['final_base'],disabled=c['disabled'],
            selected_starts=result['selected_starts'],**result['forecasts'],**{f'expert_{k}':v for k,v in values.items()}))
        if (i+1)%500==0:print('TRANSPORT TRACE',predictor.folder.name,split,i+1,'/',len(data['sid']),flush=True)
    table={k:np.stack([r[k] for r in records]) for k in records[0]};table.update({k:data[k] for k in ['raw_y','std','sid','start']})
    return table


def policies_for(val):
    values={k[7:]:v for k,v in val.items() if k.startswith('expert_')};target=(val['raw_y']-val['base'])/val['std'][:,None]
    policies={}
    for name,allowed in [('transport_only',['transport']),('transport_consensus',['transport_consensus']),('transport_routed',['encoder_only','transport','transport_consensus','transport_mix_0.25','transport_mix_0.5','transport_mix_0.75'])]:
        policies[name]=choose_policy(target,{k:values[k] for k in allowed},val['sid'],val['start'],require_mae=True)
        for sid in np.unique(val['sid']):
            if np.any(val['disabled'][val['sid']==sid]):policies[name][str(int(sid))].update(family='direct',beta=0.)
    return policies


def forecasts(table,policies):
    values={k[7:]:v for k,v in table.items() if k.startswith('expert_')}
    pred={k:table[k] for k in ['moirai_direct','no_rag','encoder_only','routed']}
    for name,policy in policies.items():pred[name]=table['final_base']+apply_policy(policy,values,table['sid'])*table['std'][:,None]
    return pred


def experiment(checkpoint,source,out):
    out.mkdir(parents=True,exist_ok=False);p=FutureStatePredictor(source/'store',checkpoint,adapter=False)
    try:
        val=collect(p,'validation');np.savez_compressed(out/'validation_trace.npz',**val)
        policies=policies_for(val);write_json(out/'policy.json',policies)
        write_json(out/'contract.json',dict(source_checkpoint=str(checkpoint),source_run_sha256=sha256(checkpoint/'run.json'),
            validation_sha256=sha256(out/'validation_trace.npz'),policy_sha256=sha256(out/'policy.json'),
            mechanism='Query prediction + observed old future - same encoder old prediction; weights and selected five unchanged',
            caveat='Previously explored test periods; all variants reported'))
        test=collect(p,'test');np.savez_compressed(out/'test_trace.npz',**test)
        with np.load(checkpoint/'predictions.npz',allow_pickle=False) as f:
            for name in ['moirai_direct','no_rag','encoder_only','routed']:np.testing.assert_allclose(test[name],f[name],atol=2e-4,rtol=2e-5)
        pred=forecasts(test,policies);truth={k:test[k] for k in ['raw_y','std','sid','start']}
        np.savez_compressed(out/'predictions.npz',**pred,**truth,selected_starts=test['selected_starts'])
        metrics={k:metric(v,truth) for k,v in pred.items()}
        result=dict(complete=True,dataset=checkpoint.name,metrics=metrics,queries=len(test['sid']),source_checkpoint=str(checkpoint),
            policy_sha256=sha256(out/'policy.json'),contract_sha256=sha256(out/'contract.json'),predictions_sha256=sha256(out/'predictions.npz'),
            vs_encoder_interval=interval(pred['encoder_only'],pred['transport_routed'],truth),
            vs_legacy_interval=interval(pred['routed'],pred['transport_routed'],truth),
            caveat='Same frozen encoder and100-to5 retrieval; development comparison, not blind')
        write_json(out/'run.json',result);print('TRANSPORT RESULT',checkpoint.name,metrics,flush=True)
    finally:p.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint-root',required=True);parser.add_argument('--source',default='v7/runs/all_datasets_20261008')
    parser.add_argument('--output',required=True);parser.add_argument('--datasets',nargs='+',required=True);a=parser.parse_args()
    for name in a.datasets:experiment(Path(a.checkpoint_root)/name,Path(a.source)/name,Path(a.output)/name)
