"""Freeze a future-state router before evaluating remaining disjoint episodes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from v5.common import valid_ranges
from .common import sha256,write_json
from .benchmark_error_correction import load
from .query_future_state_rag import FutureStatePredictor
from .benchmark_state_recall import metric,interval


def evaluate(checkpoint,source,out,exclude,count,device):
    out.mkdir(parents=True,exist_ok=False);run=json.loads((checkpoint/'run.json').read_text(encoding='utf-8'))
    old_run=json.loads((Path(run['source_checkpoint'])/'run.json').read_text(encoding='utf-8'));old=load(Path(old_run['source_cache']),'test')
    prior=[old];exclude_hashes={}
    for root in exclude:
        path=root/checkpoint.name/'predictions.npz'
        if not path.exists():raise ValueError('Missing excluded-window artifact')
        with np.load(path,allow_pickle=False) as f:prior.append(dict(sid=f['sid'].copy(),start=f['start'].copy()))
        exclude_hashes[str(path)]=sha256(path)
    store=Store(source/'store');references={}
    try:
        for s in store.series:
            sid=s['sid'];starts=np.array([p for a,b,step in valid_ranges(s,96,96,s['validation_end'],s['n'],192) for p in range(a,b,step)],dtype=np.int64)
            previous=np.concatenate([d['start'][d['sid']==sid] for d in prior])
            starts=starts[np.all(abs(starts[:,None]-previous[None,:])>=192,axis=1)]
            if len(starts)>count:starts=starts[np.linspace(0,len(starts)-1,count,dtype=int)]
            if not len(starts):raise ValueError('No remaining unseen independent episodes')
            references[str(sid)]=starts.tolist()
        write_json(out/'contract.json',dict(source_run_sha256=sha256(checkpoint/'run.json'),references=references,excluded_artifacts=exclude_hashes,
            original_test_cache=str(old_run['source_cache']),rule='No overlap with any original or excluded192-point episode; freeze before new outcomes'))
        predictor=FutureStatePredictor(source/'store',checkpoint,device);records=[]
        try:
            for s in store.series:
                sid=s['sid']
                for start in references[str(sid)]:
                    x=store.window(sid,start,96);result=predictor.predict(x,sid,start)
                    # Fetch outcome after all five forecasts have been fixed.
                    records.append(dict(**result['forecasts'],raw_y=store.window(sid,start+96,96),std=np.float32(max(s['memory_std'],1e-6)),
                        sid=np.int64(sid),start=np.int64(start),selected_starts=np.asarray(result['selected_starts'])))
                print('REMAINING WINDOWS',checkpoint.name,sid+1,'/',len(store.series),flush=True)
        finally:predictor.close()
        table={k:np.stack([r[k] for r in records]) for k in records[0]};np.savez_compressed(out/'predictions.npz',**table)
        methods=['moirai_direct','no_rag','encoder_only','future_state_rag','routed'];metrics={k:metric(table[k],table) for k in methods}
        result=dict(complete=True,dataset=checkpoint.name,variables=len(store.series),queries=len(records),metrics=metrics,
            gain_vs_direct_percent=100*(1-metrics['routed']['nmse']/metrics['moirai_direct']['nmse']),
            gain_vs_encoder_only_percent=100*(1-metrics['routed']['nmse']/metrics['encoder_only']['nmse']),
            gain_interval=interval(table['moirai_direct'],table['routed'],table),rag_interval=interval(table['encoder_only'],table['routed'],table),
            source_checkpoint=str(checkpoint),contract_sha256=sha256(out/'contract.json'),predictions_sha256=sha256(out/'predictions.npz'),
            total_previous_episode_overlap=0,caveat='Previously uninspected disjoint windows, but the same year/period was used for development')
        write_json(out/'run.json',result);print('REMAINING WINDOW RESULT',result,flush=True)
    finally:store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--output',required=True);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--exclude',nargs='+',required=True)
    p.add_argument('--count',type=int,default=24);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4)
    for name in a.datasets:evaluate(Path(a.checkpoint_root)/name,Path(a.source)/name,Path(a.output)/name,list(map(Path,a.exclude)),a.count,a.device)
