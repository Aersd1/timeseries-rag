"""Add earlier regimes to supervised training, with strictly causal retrieval."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from .common import sha256,write_json
from .moirai import MoiraiAdapter
from .trend_rag import ResidualPool
from .error_correction import retrieve
from .benchmark_error_correction import load


def augment(cache,source,out,count,device):
    out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((cache/'manifest.json').read_text(encoding='utf-8'));store=Store(source/'store')
    if meta['source_data_id']!=store.meta['data_id']:raise ValueError('Wrong data store')
    adapter=MoiraiAdapter.pretrained(meta['moirai'],device);original=load(cache,'train');rows=[]
    try:
        for s in store.series:
            sid=s['sid'];pool=ResidualPool(store,sid,meta['window'],stride=meta.get('memory_stride',1))
            path=cache/f'memory_errors_{sid}.npz'
            if sha256(path)!=meta['memory_files'][str(sid)]:raise ValueError('Changed memory cache')
            shutil.copy2(path,out/path.name)
            with np.load(path,allow_pickle=False) as f:
                np.testing.assert_array_equal(f['starts'],pool.starts);errors=f['errors'].copy()
            eligible=pool.starts[pool.starts>=pool.starts[99]+192]
            if len(eligible)<count:raise ValueError('Not enough causal prefix training episodes')
            starts=eligible[np.linspace(0,len(eligible)-1,count,dtype=int)]
            x=np.stack([store.window(sid,p,96) for p in starts]);q=adapter.forecast(x,96).cpu().numpy()
            for history,quantiles,p in zip(x,q,starts):
                _,m=retrieve(history,quantiles,pool,int(p),meta['window'])
                if np.any(m['candidate_starts']+192>p):raise ValueError('Self or overlapping retrieval')
                ids=np.searchsorted(pool.starts,m['selected_starts']);std=np.float32(max(s['memory_std'],1e-6))
                y=store.window(sid,p+96,96)
                rows.append(dict(raw_x=history,raw_quantiles=quantiles,base=quantiles[4],raw_y=y,target=(y-quantiles[4])/std,
                    std=std,sid=np.int64(sid),start=np.int64(p),errors=errors[ids],
                    **{k:m[k] for k in ['weights','selected_starts','candidate_starts','history_scores','future_scores']}))
            print(cache.name,'causal memory training',sid+1,'/',len(store.series),flush=True)
        table={k:np.concatenate([original[k],np.stack([r[k] for r in rows])]) for k in original if k in rows[0]}
        order=np.lexsort((table['start'],table['sid']));table={k:v[order] for k,v in table.items()}
        np.savez_compressed(out/'train.npz',**table)
        for split in ['validation','test']:shutil.copy2(cache/f'{split}.npz',out/f'{split}.npz')
        meta['splits']['train']=dict(queries=len(table['sid']),sha256=sha256(out/'train.npz'))
        meta['augmentation']=dict(original_cache=str(cache),original_manifest_sha256=sha256(cache/'manifest.json'),
            memory_queries_per_variable=count,rule='All100 retrieved complete192-point episodes end before each memory training query history starts; no self overlap',
            normalization='Initial-prefix statistics are available before both held-out forward folds',validation_test_byte_identical=True)
        write_json(out/'manifest.json',meta)
    finally:store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=32);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4)
    augment(Path(a.cache),Path(a.source),Path(a.output),a.count,a.device)
