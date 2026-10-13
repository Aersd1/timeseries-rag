"""History-only utility reranking against the repository's pinned Moirai2."""
import argparse
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import write_json,sha256
from .moirai import MoiraiAdapter
from .trend_rag import ResidualPool,decompose,normalize
from .error_correction import retrieve
from .benefit_rerank import pair_features,covariance,select_candidates,apply_policy


def predict(store_path,checkpoint_folder,history,sid,start,device='cuda'):
    folder=Path(checkpoint_folder);meta=json.loads((folder/'manifest.json').read_text())
    run=json.loads((folder/'run.json').read_text());policy=json.loads((folder/'policy.json').read_text())['utility']
    for path,key in [('ranker.npz','ranker_sha256'),('policy.json','policy_sha256')]:
        if sha256(folder/path)!=run[key]:raise ValueError('Checkpoint/policy changed')
    with np.load(folder/'ranker.npz',allow_pickle=False) as f:state={k:f[k].copy() for k in f.files}
    x=np.asarray(history,dtype=np.float32);store=Store(store_path)
    try:
        if x.shape!=(96,) or not np.isfinite(x).all():raise ValueError('Expected finite96 history')
        if store.meta['data_id']!=meta['source_data_id']:raise ValueError('Wrong historical store')
        if not 0<=sid<len(store.series) or start<store.series[sid]['memory_end']:raise ValueError('Query must follow initial memory')
        s=store.series[sid];std=max(float(s['memory_std']),1e-6)
        adapter=MoiraiAdapter.pretrained(dict(cache_dir='.moirai-cache',local_files_only=True),device)
        quantiles=adapter.forecast(x[None],96)[0].cpu().numpy();base=quantiles[4]
        pool=ResidualPool(store,sid,meta['window'],end=s['n'] if meta['memory_scope']=='observed' else None,
            stride=meta['memory_stride'])
        _,memory=retrieve(x,quantiles,pool,start,meta['window']);starts=memory['candidate_starts']
        ids=np.searchsorted(pool.starts,starts);past=np.stack([store.window(sid,p,96) for p in starts])
        if meta['value_mode']=='errors':
            old_pred=adapter.forecast(past,96)[:,4].cpu().numpy()
            old_truth=np.stack([store.window(sid,p+96,96) for p in starts]);values=((old_truth-old_pred)/std).astype(np.float32)
        else:
            _,r=decompose(x,meta['window']);_,mean,scale=normalize(r,pool.floor)
            _,predicted_r=decompose(np.r_[x,base],meta['window'])
            values=((pool.future[ids]*scale+mean-predicted_r[96:])/std).astype(np.float32)
        fs,hs=memory['future_scores'],memory['history_scores']
        f=pair_features(x,base,start,starts,past,pool.past[ids],values,fs,hs,s['memory_mean'],std,meta['period'],meta['window'])
        prior=.8*fs/max(float(np.median(fs)),1e-6)+.2*hs/max(float(np.median(hs)),1e-6)
        selected=select_candidates(covariance(state,f),values,prior,starts,sid)
        delta=apply_policy(policy,selected['correction'][None],[sid])[0]*std
        return dict(prediction=(base+delta).tolist(),moirai_prediction=base.tolist(),correction=delta.tolist(),
            selected_starts=starts[selected['selected']].tolist(),weights=selected['weights'].tolist(),
            beta=policy[str(sid)]['beta'],utility_permitted=selected['permitted'],length=96,horizon=96,
            memory_scope=meta['memory_scope'],value_mode=meta['value_mode'])
    finally:store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--store',required=True);p.add_argument('--checkpoint-folder',required=True)
    p.add_argument('--history',required=True);p.add_argument('--sid',type=int,required=True);p.add_argument('--start',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    write_json(a.output,predict(a.store,a.checkpoint_folder,np.load(a.history,allow_pickle=False),a.sid,a.start,a.device))
