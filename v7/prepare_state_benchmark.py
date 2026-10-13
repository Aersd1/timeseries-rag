"""Create paired raw Moirai caches for every variable of additional datasets.

No decomposition model is trained. A16-point memory indexing stride is shared
by all compared methods; query windows are sampled before any prediction.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from v5.common import valid_ranges, sample_ranges
from .common import write_json, sha256
from .moirai import MoiraiAdapter
from .trend_rag import ResidualPool
from .error_correction import retrieve


def references(series, split, count, seed):
    lo,hi = {'train':(series['memory_end'],series['train_end']),
        'validation':(series['train_end'],series['validation_end']),
        'test':(series['validation_end'],series['n'])}[split]
    ranges = list(valid_ranges(series,96,96,lo,hi,32 if split=='train' else 192))
    if split == 'train':
        return sample_ranges(ranges,count,np.random.default_rng(seed+series['sid']))
    starts = np.array([p for a,b,step in ranges for p in range(a,b,step)],dtype=np.int64)
    if len(starts)>count:
        starts=starts[np.linspace(0,len(starts)-1,count,dtype=int)]
    return starts


def prepare(source, root, name, device, train_count, eval_count, stride):
    out = root/name;out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((source/name/'config.json').read_text(encoding='utf-8'))
    store=Store(source/name/'store');adapter=MoiraiAdapter.pretrained(cfg['moirai'],device)
    shards=out/'series';shards.mkdir(exist_ok=True);memory_files={};unsupported=[]
    try:
        for s in store.series:
            sid=s['sid'];checkpoint=shards/f'{sid}.json'
            contract=dict(data_id=store.meta['data_id'],stride=stride,train_count=train_count,eval_count=eval_count,
                model_identity=adapter.identity,seed=cfg['seed'])
            if checkpoint.exists():
                saved=json.loads(checkpoint.read_text(encoding='utf-8'))
                if saved['contract']!=contract:raise ValueError('Resume contract changed')
                for path,digest in saved.get('files',{}).items():
                    if sha256(out/path)!=digest:raise ValueError('Resume file changed')
                if saved.get('unsupported'):unsupported.append(saved['unsupported'])
                else:memory_files[str(sid)]=saved['files'][f'memory_errors_{sid}.npz']
                continue
            starts={sp:references(s,sp,train_count if sp=='train' else eval_count,cfg['seed'])
                for sp in ['train','validation','test']}
            complete=sum(max(0,min(b,s['memory_end'])-a-191) for a,b in s['runs'])
            if complete<100 or len(starts['validation'])<9 or len(starts['train'])<16:
                reason=dict(sid=sid,complete_memory_episodes=complete,
                    split_queries={k:len(v) for k,v in starts.items()},
                    reason='Insufficient100-episode memory or independent validation blocks')
                unsupported.append(reason)
                # Still run the direct96-to96 reference; no fake RAG score.
                x=np.stack([store.window(sid,p,96) for p in starts['test']])
                q=adapter.forecast(x,96).cpu().numpy()
                y=np.stack([store.window(sid,p+96,96) for p in starts['test']])
                path=shards/f'{sid}_direct.npz';np.savez_compressed(path,base=q[:,4],raw_y=y,
                    start=starts['test'],sid=np.full(len(y),sid),std=max(s['memory_std'],1e-6))
                write_json(checkpoint,dict(contract=contract,unsupported=reason,
                    files={str(path.relative_to(out)):sha256(path)}))
                print(name,'unsupported',reason,flush=True);continue
            pool=ResidualPool(store,sid,25,stride=stride)
            old=[]
            for begin in range(0,len(pool.starts),128):
                p=pool.starts[begin:begin+128];x=np.stack([store.window(sid,a,96) for a in p])
                old.extend(adapter.forecast(x,96)[:,4].cpu().numpy())
            base=np.stack(old);truth=np.stack([store.window(sid,p+96,96) for p in pool.starts])
            std=np.float32(max(s['memory_std'],1e-6));errors=(truth-base)/std
            path=out/f'memory_errors_{sid}.npz'
            np.savez_compressed(path,starts=pool.starts,baseline=base,errors=errors)
            files={path.name:sha256(path)};memory_files[str(sid)]=files[path.name]
            for split,positions in starts.items():
                xs=np.stack([store.window(sid,p,96) for p in positions]);qs=[]
                for begin in range(0,len(xs),32):qs.extend(adapter.forecast(xs[begin:begin+32],96).cpu().numpy())
                arrays={}
                for x,q,start in zip(xs,qs,positions):
                    _,m=retrieve(x,q,pool,int(start),25)
                    ids=np.searchsorted(pool.starts,m['selected_starts'])
                    y=store.window(sid,start+96,96)
                    record=dict(raw_x=x,raw_quantiles=q,base=q[4],raw_y=y,target=(y-q[4])/std,
                        std=std,sid=np.int64(sid),start=np.int64(start),errors=errors[ids],
                        **{k:m[k] for k in ['weights','selected_starts','candidate_starts','history_scores','future_scores']})
                    for k,v in record.items():arrays.setdefault(k,[]).append(v)
                path=shards/f'{sid}_{split}.npz';np.savez_compressed(path,**{k:np.asarray(v) for k,v in arrays.items()})
                files[str(path.relative_to(out))]=sha256(path)
            write_json(checkpoint,dict(contract=contract,files=files))
            if sid%10==0 or sid==len(store.series)-1:print(name,'cached variables',sid+1,'/',len(store.series),flush=True)
        if unsupported:
            if len(unsupported)!=len(store.series):raise ValueError('Mixed eligible/unsupported variables require explicit handling')
            items=[dict(np.load(shards/f'{s["sid"]}_direct.npz')) for s in store.series]
            base=np.concatenate([x['base'] for x in items]);y=np.concatenate([x['raw_y'] for x in items])
            std=np.concatenate([np.repeat(x['std'],len(x['base'])) for x in items])
            write_json(out/'unsupported.json',dict(dataset=name,variables=len(store.series),queries=len(y),
                moirai_direct_nmse=float(np.mean(((base.astype(float)-y)/std[:,None])**2)),details=unsupported,
                length=96,horizon=96,reason='Original100-candidate and independent validation protocol cannot run'))
            return
        splits={}
        for split in ['train','validation','test']:
            arrays={}
            for s in store.series:
                with np.load(shards/f'{s["sid"]}_{split}.npz',allow_pickle=False) as f:
                    for k in f.files:arrays.setdefault(k,[]).append(f[k].copy())
            path=out/(split+'.npz');table={k:np.concatenate(v) for k,v in arrays.items()}
            np.savez_compressed(path,**table);splits[split]=dict(queries=len(table['sid']),sha256=sha256(path))
        write_json(out/'manifest.json',dict(complete=True,dataset=name,source_data_id=store.meta['data_id'],
            model_identity=adapter.identity,moirai=cfg['moirai'],seed=cfg['seed'],window=25,splits=splits,
            memory_files=memory_files,memory_scope='prefix',memory_stride=stride,value_mode='errors',
            train_per_series=train_count,eval_per_series=eval_count,all_variables=len(store.series),
            sampling='Train random seeded without replacement; validation/test evenly spaced nonoverlapping windows'))
    finally:store.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--output',required=True);p.add_argument('--datasets',nargs='+',default=['electricity','traffic','weather','illness'])
    p.add_argument('--device',default='cuda');p.add_argument('--train-per-series',type=int,default=32)
    p.add_argument('--eval-per-series',type=int,default=12);p.add_argument('--memory-stride',type=int,default=16)
    a=p.parse_args();torch.set_num_threads(4)
    if min(a.train_per_series,a.eval_per_series,a.memory_stride)<1:raise ValueError('Positive sampling counts and stride required')
    for name in a.datasets:prepare(Path(a.source),Path(a.output),name,a.device,a.train_per_series,a.eval_per_series,a.memory_stride)


if __name__=='__main__':main()
