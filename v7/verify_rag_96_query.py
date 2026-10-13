"""Reload all three96-point inference paths from observed history only."""
import argparse
import gc
from pathlib import Path
import numpy as np
import torch
from .common import write_json
from .query_trend_rag import predict as predict_split
from .query_error_correction import predict as predict_correction


def verify(source,split_root,error_root,online_root,device):
    results=[]
    for name,root,kind in [('ETTh1',Path(split_root),'split'),('ETTm2',Path(error_root),'errors'),('ETTm2',Path(online_root),'innovations')]:
        with np.load(root/name/'test.npz',allow_pickle=False) as d:
            ids=np.flatnonzero(d['sid']==6);i=int(ids[0]);sid,start=int(d['sid'][i]),int(d['start'][i]);x=d['raw_x'][i].copy()
            selected=d['selected_starts'][i].copy()
        checkpoint=root/name/('rag/best.pt' if kind=='split' else 'best.pt')
        result=(predict_split if kind=='split' else predict_correction)(Path(source)/name/'store',checkpoint,x,sid,start,device)
        path=root/name/('rag/predictions.npz' if kind=='split' else 'predictions.npz')
        with np.load(path,allow_pickle=False) as d: expected=d['prediction' if kind=='split' else 'rag_adopted'][i]
        np.testing.assert_allclose(result['prediction'],expected,rtol=1e-4,atol=1e-4)
        np.testing.assert_array_equal(result['selected_starts'],selected)
        record=dict(dataset=name,kind=kind,sid=sid,start=start,selected_ids_match=True,
            maximum_prediction_difference=float(np.max(abs(np.asarray(result['prediction'])-expected))),
            true_query_future_read=False)
        write_json(root/'query_verification.json',record);results.append(record)
        gc.collect()
        if device.startswith('cuda'):torch.cuda.empty_cache()
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--split-root',default='v7/runs/trend_rag_96_20261013');p.add_argument('--error-root',default='v7/runs/rag_error_correction_96_20261013')
    p.add_argument('--online-root',default='v7/runs/rag_innovation_online_96_20261013');p.add_argument('--device',default='cuda')
    a=p.parse_args();torch.set_num_threads(4);print(verify(a.source,a.split_root,a.error_root,a.online_root,a.device),flush=True)
