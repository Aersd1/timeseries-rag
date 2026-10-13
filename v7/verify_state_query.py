"""Replay a nonzero-correction test query using only observed history input."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from .common import write_json
from .query_state_recall import predict


def verify(root,source,names,device):
    records=[]
    previous=Path(root)/'query_verification.json'
    if previous.exists():
        records=[x for x in json.loads(previous.read_text(encoding='utf-8'))['records'] if x['dataset'] not in names]
    for name in names:
        folder=Path(root)/name;run=json.loads((folder/'run.json').read_text(encoding='utf-8'))
        weight=run['selected_state_weight'];branch=folder/f'weight_{weight}'
        policy=json.loads((branch/'policy.json').read_text(encoding='utf-8'))['utility']
        with np.load(branch/'test.npz',allow_pickle=False) as f:
            rows=np.flatnonzero([policy[str(int(s))]['beta']>0 for s in f['sid']])
            i=int(rows[0]) if len(rows) else 0
            sid,start=int(f['sid'][i]),int(f['start'][i]);candidates=f['starts'][i].copy()
        with np.load(branch/'predictions.npz',allow_pickle=False) as f:
            expected=f['prediction'][i].copy();selected=candidates[f['selected'][i]].tolist()
        store=Store(Path(source)/name/'store')
        try:history=store.window(sid,start,96)
        finally:store.close()
        result=predict(Path(source)/name/'store',folder,history,sid,start,device)
        # Batch-dependent floating-point kernels may cause tiny numeric changes.
        np.testing.assert_allclose(result['prediction'],expected,rtol=2e-5,atol=2e-5)
        assert result['selected_starts']==selected
        record=dict(dataset=name,sid=sid,start=start,state_weight=weight,beta=result['beta'],
            selected_starts_exact=True,history_only_input=True,
            maximum_absolute_forecast_difference=float(np.max(abs(np.asarray(result['prediction'])-expected))))
        records.append(record);print('VERIFIED STATE QUERY',record,flush=True)
    write_json(Path(root)/'query_verification.json',dict(complete=True,records=records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--datasets',nargs='+',default=['ETTm1','weather']);p.add_argument('--device',default='cuda')
    a=p.parse_args();torch.set_num_threads(4);verify(a.root,a.source,a.datasets,a.device)
