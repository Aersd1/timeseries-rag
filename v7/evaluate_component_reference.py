"""Replay a declared prior cohort with frozen variants and shared foundation outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .common import sha256,write_json
from .query_component_rag import ComponentPredictor
from .query_future_state_rag import FutureStatePredictor
from .benchmark_state_recall import metric,interval


def evaluate(component,resolution,source,references,out,device):
    out.mkdir(parents=True,exist_ok=False)
    with np.load(references,allow_pickle=False) as f:sids=f['sid'].copy();starts=f['start'].copy()
    write_json(out/'contract.json',dict(component_run_sha256=sha256(component/'run.json'),resolution_run_sha256=sha256(resolution/'run.json'),
        references_sha256=sha256(references),sid=sids.tolist(),start=starts.tolist(),
        rule='Both variants frozen; identical foundation outputs; references previously evaluated, not a blind test'))
    a=ComponentPredictor(source/'store',component,device);b=FutureStatePredictor(source/'store',resolution,adapter=a.adapter)
    try:
        histories=np.stack([a.store.window(int(sid),int(start),96) for sid,start in zip(sids,starts)])
        quantiles=[];probes=[]
        for i in range(0,len(sids),32):
            x=histories[i:i+32];quantiles.append(a.adapter.forecast(x,96).cpu().numpy());probes.append(a.adapter.forecast(x[:,:64],32).cpu().numpy())
            print('SHARED FOUNDATION',i+len(x),'/',len(sids),flush=True)
        q=np.concatenate(quantiles);probe=np.concatenate(probes);records=[]
        for i,(sid,start) in enumerate(zip(sids,starts)):
            sid,start=int(sid),int(start)
            ra=a.predict_cached(histories[i],sid,start,q[i:i+1],probe[i:i+1]);rb=b.predict_cached(histories[i],sid,start,q[i:i+1],probe[i:i+1])
            record={k:np.asarray(v) for k,v in ra['forecasts'].items()}
            record.update({f'resolution_{k}':np.asarray(rb['forecasts'][k]) for k in ['encoder_only','future_state_rag','routed']})
            # Read current outcomes only after both sets of predictions are fixed.
            record.update(raw_y=a.store.window(sid,start+96,96),sid=sid,start=start,std=np.float32(a.stds[sid]),
                selected_starts=np.asarray(ra['selected_starts']),resolution_selected_starts=np.asarray(rb['selected_starts']))
            for selected in [ra['selected_starts'],rb['selected_starts']]:assert np.all(np.asarray(selected)+192<=min(start,a.store.series[sid]['memory_end']))
            records.append(record)
            if (i+1)%100==0:print('REFERENCE REPLAY',i+1,'/',len(sids),flush=True)
        table={k:np.stack([r[k] for r in records]) for k in records[0]};np.savez_compressed(out/'predictions.npz',**table)
        np.savez_compressed(out/'foundation.npz',quantiles=q,probe=probe,sid=sids,start=starts)
        methods=list(ra['forecasts'])+[f'resolution_{k}' for k in ['encoder_only','future_state_rag','routed']]
        metrics={k:metric(table[k],table) for k in methods}
        result=dict(complete=True,queries=len(sids),metrics=metrics,contract_sha256=sha256(out/'contract.json'),
            predictions_sha256=sha256(out/'predictions.npz'),foundation_sha256=sha256(out/'foundation.npz'),
            component_checkpoint=str(component),resolution_checkpoint=str(resolution),references=str(references),
            segment_vs_legacy_interval=interval(table['routed'],table['segmented_rag'],table),
            segment_rag_gain_interval=interval(table['segmented_no_rag'],table['segmented_rag'],table),
            caveat='Same378 previously reported weather windows; exploratory paired comparison only')
        write_json(out/'run.json',result);print('REFERENCE RESULTS',result,flush=True)
    finally:a.close();b.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--component',required=True);p.add_argument('--resolution',required=True);p.add_argument('--source',required=True)
    p.add_argument('--references',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4)
    evaluate(Path(a.component),Path(a.resolution),Path(a.source),Path(a.references),Path(a.output),a.device)
