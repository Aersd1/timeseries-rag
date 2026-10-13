"""Replay actual GPU history-only inference for an adopted RAG route."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from .common import write_json
from .query_future_state_rag import FutureStatePredictor


def verify(root,source,names,evaluation_root=None):
    records=[]
    for name in names:
        folder=Path(root)/name;policies=json.loads((folder/'policy.json').read_text(encoding='utf-8'))['routed']
        eligible=[int(sid) for sid,p in policies.items() if p['family'] not in ['direct','encoder_only']]
        evaluation=Path(evaluation_root)/name if evaluation_root else folder
        with np.load(evaluation/'predictions.npz',allow_pickle=False) as f:
            ids=np.flatnonzero(np.isin(f['sid'],eligible));i=int(ids[0]) if len(ids) else 0
            sid,start=int(f['sid'][i]),int(f['start'][i]);expected={k:f[k][i].copy() for k in ['moirai_direct','no_rag','encoder_only','future_state_rag','routed']}
        store=Store(Path(source)/name/'store')
        try:history=store.window(sid,start,96)
        finally:store.close()
        predictor=FutureStatePredictor(Path(source)/name/'store',folder,'cuda')
        try:
            actual=predictor.predict(history,sid,start)
            for k,value in expected.items():np.testing.assert_allclose(actual['forecasts'][k],value,rtol=2e-5,atol=2e-5)
            records.append(dict(dataset=name,sid=sid,start=start,family=actual['family'],beta=actual['beta'],query_future_used=False,
                all_five_forecasts_verified=True,maximum_difference=max(float(np.max(abs(np.asarray(actual['forecasts'][k])-v))) for k,v in expected.items())))
            print('VERIFIED FUTURE QUERY',records[-1],flush=True)
        finally:predictor.close()
    write_json(Path(evaluation_root or root)/'query_verification.json',dict(complete=True,records=records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--evaluation-root');p.add_argument('--datasets',nargs='+',required=True);a=p.parse_args();torch.set_num_threads(4)
    verify(a.root,a.source,a.datasets,a.evaluation_root)
