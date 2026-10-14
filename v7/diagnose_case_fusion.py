"""Frozen-model controls; no test-based training, policy or hyperparameter changes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .aligned_case_fusion import inputs,tensor_inputs,forecast
from .query_case_fusion import LearnedCasePredictor
from .benchmark_case_fusion import load_trace,apply
from .benchmark_state_recall import metric,interval
from .common import write_json,sha256


def diagnose(folder,device='cuda'):
    meta=json.loads((folder/'preparation.json').read_text());run=json.loads((folder/'run.json').read_text())
    data=load_trace(folder,'test',run['test_info']['sha256'])
    p=LearnedCasePredictor(meta['store'],folder,device,adapter=False)
    try:
        normal,_,scale=inputs(data,p.stats,p.period,'wiener');rng=np.random.default_rng(20261014)
        count=data['case_past'].shape[1];order=np.stack([rng.permutation(count) for _ in data['sid']])
        shuffled={k:v.copy() for k,v in normal.items()}
        shuffled['future']=normal['future'][np.arange(len(order))[:,None],order]
        random_data=dict(data);random_starts=np.empty_like(data['selected_starts'])
        past=np.empty_like(data['case_past']);future=np.empty_like(data['case_future']);store=p.original.store
        contract=json.loads((Path(meta['source'])/'contract.json').read_text())
        for sid in np.unique(data['sid']):
            sid=int(sid);s=store.series[sid];limit=int(meta['limits'][str(sid)]);stride=contract['memory_stride']
            starts=np.array([t for a,b in s['runs'] for t in range(a,min(b,limit)-192+1,stride)])
            values=p.original.synchronous_values[sid]
            for row in np.flatnonzero(data['sid']==sid):
                eligible=starts[starts+192<=data['start'][row]]
                chosen=rng.choice(eligible,size=count,replace=False);random_starts[row]=chosen
                past[row]=values[chosen[:,None]+np.arange(96)];future[row]=values[chosen[:,None]+96+np.arange(96)]
        random_data.update(case_past=past,case_future=future,selected_starts=random_starts)
        random_features,_,random_scale=inputs(random_data,p.stats,p.period,'wiener')
        out={}
        for key,features,current_scale in [('post_alignment_shuffle',shuffled,scale),('random_cases',random_features,random_scale)]:
            features=tensor_inputs(features,device)
            raw=np.mean([forecast(m,features,data,current_scale) for m in p.models['wiener']],axis=0).astype(np.float32)
            out['raw_'+key]=raw;out['calibrated_'+key]=apply(p.policies['wiener'],data,raw)
            if count==5 and not meta.get('residual_reference'):
                with np.load(folder/'predictions.npz',allow_pickle=False) as f:incumbent=f['previous_previous_five'].copy()
                proposal=incumbent+raw-data['raw_quantiles'][:,4]
                out['increment_'+key]=apply(p.policies['increment_wiener'],data,proposal,incumbent)
        with np.load(folder/'predictions.npz',allow_pickle=False) as f:
            base=f['calibrated_wiener'].copy();rawbase=f['raw_wiener'].copy()
        np.savez_compressed(folder/'controls.npz',**out,random_starts=random_starts,shuffle_order=order)
        result=dict(complete=True,queries=len(data['sid']),metrics={k:metric(v,data) for k,v in out.items()},
            shuffled_vs_real_interval=interval(out['calibrated_post_alignment_shuffle'],base,data),
            random_vs_real_interval=interval(out['calibrated_random_cases'],base,data),
            raw_random_vs_real_interval=interval(out['raw_random_cases'],rawbase,data),
            file_sha256=sha256(folder/'controls.npz'),model_contract_sha256=run['contract_sha256'],
            caveat='Inference stress controls with fixed trained model; cannot alone establish causal effects or benefits of retraining with random retrieval')
        write_json(folder/'controls.json',result);print('FUSION CONTROLS',folder.name,result['metrics'],flush=True)
    finally:p.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--datasets',nargs='+',required=True);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();torch.set_num_threads(4)
    for name in args.datasets:diagnose(Path(args.root)/name,args.device)
