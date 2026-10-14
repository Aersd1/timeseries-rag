"""Freeze a guarded RAG policy before paired reference or shifted-target evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .query_guarded_bank import GuardedBankPredictor
from .common import sha256,write_json
from .benchmark_state_recall import metric,interval


def evaluate(checkpoint,source,references,out,foundation=None,offset=0,device='cuda'):
    out.mkdir(parents=True,exist_ok=False)
    with np.load(references,allow_pickle=False) as f:sids=f['sid'].copy();prior_starts=f['start'].copy()
    prior_sids=sids.copy()
    p=GuardedBankPredictor(source/'store',checkpoint,device,adapter=False if foundation else None)
    try:
        starts=prior_starts+offset;keep=[]
        for i,(sid,start) in enumerate(zip(sids,starts)):
            s=p.original.store.series[int(sid)]
            if start>=s['validation_end'] and start+192<=s['n'] and any(a<=start and start+192<=b for a,b in s['runs']):keep.append(i)
        sids=sids[keep];starts=starts[keep]
        if offset:
            for sid in np.unique(sids):
                new=starts[sids==sid];old=prior_starts[prior_sids==sid]
                if not np.all(abs(new[:,None]-old[None,:])>=96):raise ValueError('Shifted forecast targets overlap prior target labels')
        write_json(out/'contract.json',dict(source_run_sha256=sha256(checkpoint/'run.json'),references_sha256=sha256(references),
            foundation_sha256=sha256(foundation) if foundation else None,sid=sids.tolist(),start=starts.tolist(),offset=offset,
            rule='Freeze policies and references before paired outcomes; shifted forecast targets disjoint from original target labels',
            caveat='Previously developed year/period; shifted episodes overlap prior contexts; not a blind new dataset'))
        histories=np.stack([p.original.store.window(int(sid),int(start),96) for sid,start in zip(sids,starts)])
        if foundation:
            if offset:raise ValueError('Cannot reuse foundation outputs for shifted inputs')
            with np.load(foundation,allow_pickle=False) as f:
                np.testing.assert_array_equal(f['sid'][keep],sids);np.testing.assert_array_equal(f['start'][keep],starts)
                q=f['quantiles'][keep].copy();probe=f['probe'][keep].copy()
        else:
            quantiles=[];probes=[]
            for i in range(0,len(sids),32):
                x=histories[i:i+32];quantiles.append(p.original.adapter.forecast(x,96).cpu().numpy());probes.append(p.original.adapter.forecast(x[:,:64],32).cpu().numpy())
                print('COHORT FOUNDATION',checkpoint.name,i+len(x),'/',len(sids),flush=True)
            q=np.concatenate(quantiles);probe=np.concatenate(probes)
        records=[]
        for i,(sid,start) in enumerate(zip(sids,starts)):
            sid,start=int(sid),int(start);result=p.predict_cached(histories[i],sid,start,q[i:i+1],probe[i:i+1])
            records.append(dict(**result['forecasts'],raw_y=p.original.store.window(sid,start+96,96),std=np.float32(p.original.stds[sid]),
                sid=sid,start=start,selected_starts=np.asarray(result['selected_starts'])))
            if (i+1)%100==0:print('COHORT PREDICTIONS',checkpoint.name,i+1,'/',len(sids),flush=True)
        table={k:np.stack([r[k] for r in records]) for k in records[0]};np.savez_compressed(out/'predictions.npz',**table)
        np.savez_compressed(out/'foundation.npz',quantiles=q,probe=probe,sid=sids,start=starts)
        methods=list(result['forecasts']);metrics={k:metric(table[k],table) for k in methods}
        run=dict(complete=True,queries=len(sids),metrics=metrics,source_checkpoint=str(checkpoint),references=str(references),
            contract_sha256=sha256(out/'contract.json'),predictions_sha256=sha256(out/'predictions.npz'),foundation_sha256=sha256(out/'foundation.npz'),
            gain_interval=interval(table['routed'],table['guarded_rag'],table),rag_interval=interval(table['guarded_no_rag'],table['guarded_rag'],table),
            caveat='Paired development evaluation; reference reuse or new target slots in a previously developed period')
        write_json(out/'run.json',run);print('GUARDED COHORT RESULT',checkpoint.name,run,flush=True)
    finally:p.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--source',required=True);p.add_argument('--references',required=True);p.add_argument('--output',required=True)
    p.add_argument('--foundation');p.add_argument('--offset',type=int,default=0);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4)
    evaluate(Path(a.checkpoint),Path(a.source),Path(a.references),Path(a.output),Path(a.foundation) if a.foundation else None,a.offset,a.device)
