"""History-only raw Moirai plus bounded trained residual adaptation."""
import argparse
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import sha256,write_json
from .moirai import MoiraiAdapter,MODEL_ID,MODEL_REVISION
from .trend_rag import ResidualPool
from .error_correction import retrieve
from .residual_adapter import features
from .benefit_rerank import covariance
from .safe_error_policy import bound_delta,apply_policy


def predict(store_path,cache_path,checkpoint_folder,history,sid,start,device='cuda',use_rag=True):
    folder,cache=Path(checkpoint_folder),Path(cache_path)
    run=json.loads((folder/'run.json').read_text(encoding='utf-8'));meta=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    if sha256(cache/'manifest.json')!=run['source_manifest_sha256']:raise ValueError('Changed source cache')
    if meta['model_identity']['model_id']!=MODEL_ID or meta['model_identity']['revision']!=MODEL_REVISION:raise ValueError('Wrong pinned model')
    if sha256(folder/'policy.json')!=run['policy_sha256']:raise ValueError('Changed policy')
    kind='with_rag' if use_rag else 'without_rag';name=f'adapter_{sid}_{kind}.npz'
    if sha256(folder/name)!=run['adapter_hashes'][name]:raise ValueError('Changed adapter weights')
    with np.load(folder/name,allow_pickle=False) as f:state={k:f[k].copy() for k in f.files}
    policy=json.loads((folder/'policy.json').read_text(encoding='utf-8'))[kind];store=Store(store_path)
    try:
        if store.meta['data_id']!=run['source_data_id']:raise ValueError('Wrong store')
        s=store.series[sid]
        if start<s['validation_end']:raise ValueError('Policy validation outcomes must all be observed before the query')
        x=np.asarray(history,dtype=np.float32)
        if x.shape!=(96,) or not np.isfinite(x).all():raise ValueError('Expected finite96 history')
        adapter=MoiraiAdapter.pretrained(dict(cache_dir='.moirai-cache',local_files_only=True),device)
        quantiles=adapter.forecast(x[None],96)[0].cpu().numpy();base=quantiles[4]
        pool=ResidualPool(store,sid,25,stride=meta.get('memory_stride',1))
        _,memory=retrieve(x,quantiles,pool,start,25);selected=memory['selected_starts']
        if np.any(selected+192>start):raise ValueError('Overlapping retrieval')
        path=cache/f'memory_errors_{sid}.npz'
        if sha256(path)!=meta['memory_files'][str(sid)]:raise ValueError('Changed historical error cache')
        with np.load(path,allow_pickle=False) as f:
            np.testing.assert_array_equal(pool.starts,f['starts']);errors=f['errors'][np.searchsorted(pool.starts,selected)]
        rag=np.sum(memory['weights'][:,None]*errors,axis=0);std=np.float32(max(s['memory_std'],1e-6))
        f=features(x[None],base[None],quantiles[None],rag[None],np.array([start]),np.array([std]),
            np.array([s['memory_mean']]),run['period'],use_rag)
        delta=bound_delta(covariance(state,f)*std,quantiles[None])/std
        final=base+apply_policy(policy,{kind:delta},np.array([sid]))[0]*std
        return dict(prediction=final.tolist(),moirai_prediction=base.tolist(),selected_starts=selected.tolist(),
            beta=policy[str(sid)]['beta'],family=policy[str(sid)]['family'],length=96,horizon=96,
            use_rag=use_rag,foundation_frozen=True)
    finally:store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--store',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--checkpoint-folder',required=True);p.add_argument('--history',required=True)
    p.add_argument('--sid',type=int,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--without-rag',action='store_true');a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    write_json(a.output,predict(a.store,a.cache,a.checkpoint_folder,np.load(a.history,allow_pickle=False),a.sid,a.start,a.device,not a.without_rag))
