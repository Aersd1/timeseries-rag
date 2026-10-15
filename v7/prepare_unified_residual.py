"""Reuse pinned Moirai forecasts; prepare causal real/random/shuffled 5+5 inputs."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from .common import sha256,write_json
from .moirai import MoiraiAdapter
from .query_guarded_bank import GuardedBankPredictor
from .unified_residual import matched_starts,state_features


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def initialize(source,out,device='cuda'):
    if (out/'preparation.json').exists():
        saved=read(out/'preparation.json')
        if Path(saved['source']).resolve()!=source.resolve():raise ValueError('Resume source differs from frozen preparation')
        if sha256(source/'preparation.json')!=saved['source_preparation_sha256'] or sha256(out/'stats.json')!=saved['stats_sha256']:raise ValueError('Changed preparation inputs')
        for split in ['train','validation']:
            if not (out/f'{split}_info.json').exists():prepare_split(out,split)
        return
    out.mkdir(parents=True,exist_ok=True);(out/'banks').mkdir(exist_ok=True)
    meta=read(source/'preparation.json');cache=Path(meta['source_cache']);manifest=read(cache/'manifest.json')
    if sha256(cache/'manifest.json')!=meta['cache_manifest_sha256']:raise ValueError('Changed source cache')
    store=Store(meta['store']);adapter=None;banks={};reused=0;generated=0
    try:
        stride=read(Path(meta['source'])/'contract.json')['memory_stride']
        for s in store.series:
            sid=s['sid'];path=out/'banks'/f'{sid}.npz';end=int(meta['limits'][str(sid)])
            if path.exists():
                with np.load(path) as f:
                    if int(f['end'])!=end:raise ValueError('Changed bank limit')
                banks[str(sid)]=sha256(path);continue
            starts=np.array([p for a,b in s['runs'] for p in range(a,min(b,end)-192+1,stride)],dtype=np.int64)
            values=store.values(sid);past=np.asarray(values[starts[:,None]+np.arange(96)],dtype=np.float32)
            baseline=np.zeros_like(past);done=np.zeros(len(starts),dtype=bool)
            old=cache/f'memory_errors_{sid}.npz';expected=manifest.get('memory_files',{}).get(str(sid))
            if expected:
                if sha256(old)!=expected:raise ValueError('Changed historical Moirai cache')
                with np.load(old,allow_pickle=False) as f:
                    pos=np.searchsorted(f['starts'],starts);ok=pos<len(f['starts'])
                    ok[ok]&=f['starts'][pos[ok]]==starts[ok]
                    baseline[ok]=f['baseline'][pos[ok]];done=ok;reused+=int(ok.sum())
            missing=np.flatnonzero(~done)
            if len(missing) and adapter is None:adapter=MoiraiAdapter.pretrained(manifest['moirai'],device)
            for begin in range(0,len(missing),256):
                ids=missing[begin:begin+256];baseline[ids]=adapter.forecast(past[ids],96)[:,4].cpu().numpy()
            generated+=len(missing)
            if not np.isfinite(baseline).all():raise ValueError('Nonfinite case forecast')
            np.savez_compressed(path,starts=starts,past=past,baseline=baseline,end=np.array(end))
            banks[str(sid)]=sha256(path)
            if sid%50==0:print('UNIFIED BANK',out.name,sid,'reused',reused,'generated',generated,flush=True)
        previous=GuardedBankPredictor(meta['store'],read(Path(meta['source'])/'contract.json')['source'],adapter=False)
        try:probes=Path(previous.original.base_run['source_probes'])
        finally:previous.close()
        stats=read(source/'stats.json');stats['memory_std']=[max(s['memory_std'],1e-6) for s in store.series]
        stats['memory_mean']=[s['memory_mean'] for s in store.series]
        stats['case_count']=10;write_json(out/'stats.json',stats)
        write_json(out/'preparation.json',dict(source=str(source),source_preparation_sha256=sha256(source/'preparation.json'),
            source_run_sha256=sha256(source/'run.json'),source_data_id=meta['source_data_id'],store=meta['store'],
            period=meta['period'],limits=meta['limits'],source_cache=str(cache),source_manifest_sha256=sha256(cache/'manifest.json'),
            probes=str(probes),probe_manifest_sha256=sha256(probes/'manifest.json'),banks=banks,stats_sha256=sha256(out/'stats.json'),
            reference='Identical frozen pretrained Moirai96-to96 median for query and old cases; no dataset-fit reference',
            retrieval='Exact existing ten selected cases; source retriever was TRAIN-fitted and is NOT forward OOF inside TRAIN',
            reused_forecasts=reused,generated_forecasts=generated))
        for split in ['train','validation']:prepare_split(out,split)
    finally:store.close()


def prepare_split(out,split):
    meta=read(out/'preparation.json');source=Path(meta['source']);original=read(source/'preparation.json')
    if split=='test' and not (out/'contract.json').exists():raise ValueError('Freeze all model/policy arms before TEST')
    expected=read(source/'run.json')['test_info']['sha256'] if split=='test' else original['files'][split]['sha256']
    if sha256(source/f'{split}.npz')!=expected:raise ValueError('Changed source cases')
    with np.load(source/f'{split}.npz',allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    probes=Path(meta['probes']);pm=read(probes/'manifest.json')
    if sha256(probes/'manifest.json')!=meta['probe_manifest_sha256'] or sha256(probes/f'{split}.npz')!=pm['files'][split]:raise ValueError('Changed observed probe')
    with np.load(probes/f'{split}.npz',allow_pickle=False) as f:
        for k in ['sid','start']:np.testing.assert_array_equal(data[k],f[k])
        data['probe_quantiles']=f['quantiles'].copy()
    for k in ['case_baseline','random_case_past','random_case_future','random_case_baseline']:data[k]=np.empty_like(data['case_past'])
    data['random_selected_starts']=np.empty_like(data['selected_starts']);order=np.empty_like(data['selected_starts'])
    store=Store(meta['store']);stats=read(out/'stats.json')
    try:
        for sid in np.unique(data['sid']):
            sid=int(sid);path=out/'banks'/f'{sid}.npz'
            if sha256(path)!=meta['banks'][str(sid)]:raise ValueError('Changed bank')
            with np.load(path,allow_pickle=False) as saved_bank:bank={k:saved_bank[k].copy() for k in saved_bank.files}
            raw=store.values(sid)
            bf=state_features(bank['past'],bank['starts'],meta['period'],stats['memory_mean'][sid],stats['memory_std'][sid])
            for row in np.flatnonzero(data['sid']==sid):
                start=int(data['start'][row]);selected=data['selected_starts'][row]
                if np.any(selected+192>min(start,int(bank['end']))):raise ValueError('Overlapping old case')
                ids=np.searchsorted(bank['starts'],selected);np.testing.assert_array_equal(bank['starts'][ids],selected)
                data['case_baseline'][row]=bank['baseline'][ids]
                seed=20261015+sid*1000003+start
                matched=matched_starts(bank['starts'],bank['past'],selected,start,meta['period'],stats['memory_mean'][sid],stats['memory_std'][sid],seed,bf)
                ids=np.searchsorted(bank['starts'],matched);data['random_selected_starts'][row]=matched
                data['random_case_past'][row]=bank['past'][ids];data['random_case_baseline'][row]=bank['baseline'][ids]
                data['random_case_future'][row]=raw[matched[:,None]+96+np.arange(96)]
                rng=np.random.default_rng(seed+17)
                for j in [0,5]:
                    perm=rng.permutation(5)
                    if np.array_equal(perm,np.arange(5)):perm=np.roll(perm,1)
                    order[row,j:j+5]=j+perm
        data['shuffle_order']=order
        path=out/f'{split}.npz';np.savez_compressed(path,**data)
        result=dict(queries=len(data['sid']),sha256=sha256(path),source_sha256=expected,case_pairs=len(data['sid'])*10)
        write_json(out/f'{split}_info.json',result);print('UNIFIED PREPARED',out.name,split,len(data['sid']),flush=True)
        return result
    finally:store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4)
    for name in a.datasets:initialize(Path(a.source)/name,Path(a.output)/name,a.device)
