"""Replay an adopted past-error-adapter query against the cached paired evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from .common import write_json
from .query_past_error_adapter import predict


def verify(root,source,names,evaluation_root=None):
    records=[]
    for name in names:
        folder=Path(root)/name;run=json.loads((folder/'run.json').read_text(encoding='utf-8'))
        evaluation=Path(evaluation_root)/name if evaluation_root else folder
        with np.load(evaluation/'predictions.npz',allow_pickle=False) as f:
            eligible=np.flatnonzero(np.isin(f['sid'],run['adopted_variables']['with_rag']));i=int(eligible[0]) if len(eligible) else 0
            sid,start=int(f['sid'][i]),int(f['start'][i]);expected=f['past_with_rag'][i].copy()
        store=Store(Path(source)/name/'store')
        try:x=store.window(sid,start,96)
        finally:store.close()
        r=predict(Path(source)/name/'store',run['source_cache'],folder,x,sid,start,'cuda')
        np.testing.assert_allclose(r['prediction'],expected,rtol=2e-5,atol=2e-5)
        records.append(dict(dataset=name,sid=sid,start=start,beta=r['beta'],history_only_input=True,
            maximum_forecast_difference=float(np.max(abs(np.asarray(r['prediction'])-expected)))))
        print('VERIFIED PAST ERROR QUERY',records[-1],flush=True)
    write_json(Path(evaluation_root or root)/'query_verification.json',dict(complete=True,records=records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--evaluation-root');p.add_argument('--datasets',nargs='+',default=['weather','electricity']);a=p.parse_args();torch.set_num_threads(4)
    verify(a.root,a.source,a.datasets,a.evaluation_root)
