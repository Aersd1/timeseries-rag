"""Live state-aware retrieval: observed96-point query; no query target input."""
import argparse
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import sha256,write_json
from .moirai import MoiraiAdapter
from .trend_rag import ResidualPool
from .state_recall import StateRecall
from .benefit_rerank import pair_features,covariance,select_candidates,apply_policy


def predict(store_path,checkpoint_folder,history,sid,start,device='cuda'):
    folder=Path(checkpoint_folder);run=json.loads((folder/'run.json').read_text(encoding='utf-8'))
    weight=run['selected_state_weight'];method='utility_shape' if weight==0 else 'utility_state'
    branch=folder/f'weight_{weight}';saved=run['methods'][method]
    for name,key in [('ranker.npz','ranker_sha256'),('policy.json','policy_sha256')]:
        if sha256(branch/name)!=saved[key]:raise ValueError('Checkpoint changed')
    with np.load(branch/'ranker.npz',allow_pickle=False) as f:state={k:f[k].copy() for k in f.files}
    policy=json.loads((branch/'policy.json').read_text(encoding='utf-8'))['utility'];store=Store(store_path)
    try:
        if store.meta['data_id']!=run['source_data_id']:raise ValueError('Store mismatch')
        x=np.asarray(history,dtype=np.float32)
        if x.shape!=(96,) or not np.isfinite(x).all():raise ValueError('Expected finite96 history')
        if not 0<=sid<len(store.series) or start<store.series[sid]['memory_end']:raise ValueError('Query precedes memory endpoint')
        s=store.series[sid];std=max(s['memory_std'],1e-6)
        adapter=MoiraiAdapter.pretrained(dict(cache_dir='.moirai-cache',local_files_only=True),device)
        quantiles=adapter.forecast(x[None],96)[0].cpu().numpy();base=quantiles[4]
        pool=ResidualPool(store,sid,run['window'],stride=run['memory_stride'])
        past=np.stack([store.window(sid,p,96) for p in pool.starts])
        memory=StateRecall(pool,past,std,run['period']).retrieve(x,quantiles,start,weight,run['window'])
        starts=memory['candidate_starts'];ids=np.searchsorted(pool.starts,starts);old_past=past[ids]
        old_prediction=adapter.forecast(old_past,96)[:,4].cpu().numpy()
        old_truth=np.stack([store.window(sid,p+96,96) for p in starts])
        values=((old_truth-old_prediction)/std).astype(np.float32)
        f=pair_features(x,base,start,starts,old_past,pool.past[ids],values,memory['future_scores'],memory['history_scores'],
            s['memory_mean'],std,run['period'] or 1,run['window'])
        selected=select_candidates(covariance(state,f),values,memory['prior'],starts,sid)
        correction=apply_policy(policy,selected['correction'][None],[sid])[0]*std
        return dict(prediction=(base+correction).tolist(),moirai_prediction=base.tolist(),correction=correction.tolist(),
            selected_starts=starts[selected['selected']].tolist(),weights=selected['weights'].tolist(),
            candidate_starts=starts.tolist(),state_weight=weight,beta=policy[str(sid)]['beta'],
            utility_permitted=selected['permitted'],length=96,horizon=96)
    finally:store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--store',required=True);p.add_argument('--checkpoint-folder',required=True)
    p.add_argument('--history',required=True);p.add_argument('--sid',type=int,required=True);p.add_argument('--start',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    write_json(a.output,predict(a.store,a.checkpoint_folder,np.load(a.history,allow_pickle=False),a.sid,a.start,a.device))
