"""Replay a fixed observed-history query after utility checkpoint reload."""
import argparse
from pathlib import Path
import numpy as np
from .common import write_json
from .query_benefit_rerank import predict


def verify(root,source,cache,name,sid,device):
    folder=Path(root)/cache/name
    with np.load(folder/'test.npz',allow_pickle=False) as f:
        ids=np.flatnonzero(f['sid']==sid);i=int(ids[0]);start=int(f['start'][i]);starts=f['starts'][i].copy()
    from v5.data import Store
    store=Store(Path(source)/name/'store')
    try:history=store.window(sid,start,96)
    finally:store.close()
    result=predict(Path(source)/name/'store',folder,history,sid,start,device)
    with np.load(folder/'predictions.npz',allow_pickle=False) as f:
        expected=f['utility_stable'][i];selected=starts[f['selected'][i]]
    np.testing.assert_allclose(result['prediction'],expected,rtol=1e-4,atol=1e-4)
    np.testing.assert_array_equal(result['selected_starts'],selected)
    record=dict(dataset=name,sid=sid,start=start,selected_ids_match=True,beta=result['beta'],
        maximum_prediction_difference=float(np.max(abs(np.asarray(result['prediction'])-expected))),
        true_query_future_read=False)
    write_json(folder/'query_verification.json',record);print(record,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--cache',default='rag_error_correction_96_20261013');p.add_argument('--dataset',default='ETTm1')
    p.add_argument('--sid',type=int,default=0);p.add_argument('--device',default='cuda');a=p.parse_args()
    import torch
    torch.set_num_threads(4);verify(a.root,a.source,a.cache,a.dataset,a.sid,a.device)
