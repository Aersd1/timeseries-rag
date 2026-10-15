"""Prefix-refitted 5+5 retrieval and reusable history-only Moirai forecasts."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from .common import sha256,write_json
from .moirai import MoiraiAdapter
from .trend_rag import ResidualPool,decompose,normalize
from .dual_five_rag import RawHistoryBank
from .conditional_recall import recall
from .future_state_encoder import history_features,future_state,prototype_state,fit,predict
from .unified_residual import matched_starts,state_features

PLAN=dict(length=96,horizon=96,query_stride=96,fold_fractions=[.35,.60,.80,1.],
    expert_origins=64,selector_origins=128,first_expert_origin=2304,group_size=4,
    state_penalty=10.,state_history_width=8,state_future_width=16,
    candidates=100,case_count=10,history_weight=.2,future_weight=.8,raw_spacing=192,
    scope='Initial memory prefix, capped by each fit cutoff; no VAL-selected memory scope')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sample(values,count):
    a=np.asarray(values,dtype=np.int64)
    return a if len(a)<=count else a[np.linspace(0,len(a)-1,count,dtype=int)]


def complete_origins(series,starts):
    return np.array([int(p) for p in starts if any(a<=p and p+192<=b for a,b in series['runs'])],dtype=np.int64)


def layout(store):
    end=min(s['train_end'] for s in store.series);validation=min(s['validation_end'] for s in store.series)
    cuts=[int(np.ceil(end*f/96))*96 for f in PLAN['fold_fractions'][:-1]]+[end]
    if cuts[0]<PLAN['first_expert_origin']+384:raise ValueError('Insufficient prefix for the common protocol')
    oof=sample([p for p in range(cuts[0],end-191,96) if any(a<=p and p+192<=b for a,b in zip(cuts[:-1],cuts[1:]))],PLAN['selector_origins'])
    return dict(cuts=cuts,oof_starts=oof.tolist(),validation_starts=list(range(end,validation-191,96)),
        expert_starts={str(c):sample(np.arange(PLAN['first_expert_origin'],c-191,96),PLAN['expert_origins']).tolist() for c in cuts},
        train_end=end,validation_end=validation)


def initialize(source,out):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    meta=read(source/'preparation.json');store=Store(meta['store'])
    try:
        plan=layout(store);cache=read(Path(meta['source_cache'])/'manifest.json')
        record=dict(source=str(source.resolve()),store=str(Path(meta['store']).resolve()),data_id=store.meta['data_id'],
            catalog_sha256=sha256(Path(meta['store'])/'catalog.json'),source_preparation_sha256=sha256(source/'preparation.json'),
            moirai=cache['moirai'],period=meta['period'],plan=PLAN,layout=plan,
            caveat='New prefix-refitted upstream pipeline; not byte-identical old cases. Previously explored TEST remains exploratory.')
        if (out/'preparation.json').exists():
            if read(out/'preparation.json')!=record:raise ValueError('Changed preparation protocol')
        else:write_json(out/'preparation.json',record)
        (out/'foundation').mkdir(exist_ok=True)
        return record
    finally:store.close()


def foundation(source,out,device='cuda',test=False):
    out=Path(out);meta=initialize(source,out);store=Store(meta['store']);adapter=None
    if test and not (out/'selector_contract.json').exists():raise ValueError('Freeze all selector policies before TEST forecasts')
    namespace='foundation_test' if test else 'foundation';(out/namespace).mkdir(exist_ok=True)
    manifest_file=out/('test_foundation_manifest.json' if test else 'foundation_manifest.json')
    # Read prior TRAIN/VAL predictions only while preparing developmental data.
    reuse={}
    for split in (['test'] if test else ['train','validation']):
        if sha256(Path(source)/(split+'.npz'))!=read(Path(source)/(split+'_info.json'))['sha256']:raise ValueError('Changed upstream foundation query cache')
        with np.load(Path(source)/(split+'.npz'),allow_pickle=False) as f:
            sids,positions,qs,probes=[f[k] for k in ['sid','start','raw_quantiles','probe_quantiles']]
            for row,(sid,start) in enumerate(zip(sids,positions)):
                reuse[(int(sid),int(start))]=(qs[row].copy(),probes[row].copy())
    try:
        manifest=read(manifest_file) if manifest_file.exists() else dict(files={},reference=meta['moirai'])
        for s in store.series:
            sid=s['sid'];file=out/namespace/f'{sid}.npz'
            starts=set(meta['layout']['oof_starts'])|set(meta['layout']['validation_starts'])
            for values in meta['layout']['expert_starts'].values():starts.update(values)
            legacy=[]
            if test:
                with np.load(Path(source)/'test.npz',allow_pickle=False) as f:legacy=f['start'][f['sid']==sid].tolist()
                starts=set(range(s['validation_end'],s['n']-191,96))|set(legacy)
            requested=complete_origins(s,sorted(starts))
            saved={}
            if file.exists():
                if manifest['files'].get(str(sid))!=sha256(file):raise ValueError('Changed foundation cache')
                with np.load(file,allow_pickle=False) as f:
                    for p,q,pr in zip(f['starts'],f['quantiles'],f['probe']):saved[int(p)]=(q.copy(),pr.copy())
            missing=[int(p) for p in requested if int(p) not in saved]
            generate=[];reused=0
            for p in missing:
                if (sid,p) in reuse:saved[p]=reuse[(sid,p)];reused+=1
                else:generate.append(p)
            if generate and adapter is None:adapter=MoiraiAdapter.pretrained(meta['moirai'],device)
            for begin in range(0,len(generate),256):
                positions=generate[begin:begin+256];x=np.stack([store.window(sid,p,96) for p in positions])
                q=adapter.forecast(x,96).cpu().numpy();pr=adapter.forecast(x[:,:64],32).cpu().numpy()
                for p,a,b in zip(positions,q,pr):saved[p]=(a,b)
            if missing:
                order=np.array(sorted(saved),dtype=np.int64)
                np.savez_compressed(file,starts=order,quantiles=np.stack([saved[int(p)][0] for p in order]),probe=np.stack([saved[int(p)][1] for p in order]))
                manifest['files'][str(sid)]=sha256(file);write_json(manifest_file,manifest)
            if sid%25==0:print('FORWARD FOUNDATION',out.name,'TEST' if test else 'TRAIN_VAL',sid,'/',len(store.series),'new',len(generate),'reused',reused,flush=True)
    finally:store.close()


class PrefixStore:
    def __init__(self,store,cutoff,means,stds):
        self.original=store
        self.series=[dict(s,memory_end=min(s['memory_end'],cutoff),train_end=cutoff,
            memory_mean=float(means[s['sid']]),memory_std=float(stds[s['sid']])) for s in store.series]
    def window(self,*a):return self.original.window(*a)


def prefix_state(store,cutoff):
    means=[];stds=[];down=[]
    for s in store.series:
        x=np.asarray(store.values(s['sid'])[:cutoff],dtype=float);mean=float(np.nanmean(x));sd=max(float(np.nanstd(x)),1e-6)
        means.append(mean);stds.append(sd);down.append(np.nan_to_num((x[::16]-mean)/sd,nan=0.))
    z=np.stack(down,1);norm=np.maximum(np.linalg.norm(z,axis=0),1e-8);corr=(z.T@z)/norm[:,None]/norm[None]
    groups={}
    for sid in range(len(means)):
        order=np.lexsort((np.arange(len(means)),-abs(corr[sid])))
        groups[str(sid)]=[sid]+[int(j) for j in order if j!=sid][:PLAN['group_size']-1]
    return dict(mean=means,std=stds,memory_mean=means,memory_std=stds,case_count=10,groups=groups,cutoff=cutoff)


class PrefixRetriever:
    def __init__(self,store,stats,sid,period,stride):
        self.store=store;self.stats=stats;self.sid=sid;self.period=period;self.cutoff=stats['cutoff']
        self.prefix=PrefixStore(store,self.cutoff,stats['mean'],stats['std']);end=self.prefix.series[sid]['memory_end']
        self.pool=ResidualPool(self.prefix,sid,25,end=end,stride=stride);self.raw=RawHistoryBank(self.prefix,sid,end,stride,192)
        self.group=stats['groups'][str(sid)];starts=np.arange(0,self.cutoff-191,96,dtype=np.int64)
        def joint(positions,length):
            return np.stack([np.asarray(store.values(j))[positions[:,None]+np.arange(length)] for j in self.group],2)
        episodes=joint(starts,192);ok=np.isfinite(episodes).all((1,2));starts=sample(starts[ok],128)
        episodes=joint(starts,192)
        if len(starts)<16:
            self.group=[sid];episodes=joint(np.arange(0,self.cutoff-191,96,dtype=np.int64),192)
            starts=np.arange(0,self.cutoff-191,96,dtype=np.int64)[np.isfinite(episodes).all((1,2))];starts=sample(starts,128);episodes=joint(starts,192)
        if len(starts)<16:raise ValueError('Insufficient prefix state training')
        mm=np.asarray(stats['mean'])[self.group];sd=np.asarray(stats['std'])[self.group]
        xf=history_features(episodes[:,:96],starts,mm,sd,period or 1,True,8);y=future_state(episodes[:,:96],episodes[:,96:],sd,16)
        self.encoder=fit(xf,y,PLAN['state_penalty']);error=np.mean((predict(self.encoder,xf)-y)**2,0)
        precision=1/np.maximum(error,.1);self.encoder['precision']=np.clip(precision/max(float(np.median(precision)),1e-6),.05,20.)
        old=joint(self.pool.starts,192);self.prototypes=prototype_state(old[:,:96],old[:,96:],mm,sd,16)
        self.mm=mm;self.sd=sd
        self.random_features=state_features(self.raw.past,self.raw.starts,period,stats['mean'][sid],stats['std'][sid])

    def select(self,x,start,quantiles):
        # Sibling channels may be missing even when the queried channel is valid.
        # history_features masks/imputes only these OBSERVED history values.
        joint=np.stack([np.asarray(self.store.values(j)[start:start+96],dtype=np.float32) for j in self.group],1)
        if joint.shape!=(96,len(self.group)):raise ValueError('Incomplete observed group history')
        joint[:,self.group.index(self.sid)]=x
        code=predict(self.encoder,history_features(joint[None],np.array([start]),self.mm,self.sd,self.period or 1,True,8))[0]
        _,r=decompose(x,25);qr,mean,scale=normalize(r,self.pool.floor)
        paths=np.concatenate([np.repeat(x[None],9,0),quantiles],1);_,rr=decompose(paths,25);rq=(rr[:,96:]-mean)/scale
        m=recall(self.pool,qr,rq,code,self.prototypes,self.encoder['precision'],start)
        raw,_=self.raw.select(x,start,m['selected_starts']);chosen=np.r_[m['selected_starts'],raw].astype(np.int64)
        random=matched_starts(self.raw.starts,self.raw.past,chosen,start,self.period,self.stats['mean'][self.sid],self.stats['std'][self.sid],
            20261015+self.sid*1000003+int(start),self.random_features)
        if len(np.unique(chosen))!=10 or np.any(chosen+192>min(start,self.raw.end)):raise ValueError('Invalid causal 5+5 cases')
        return chosen,random


def make_index(source,out,cutoff,starts_by_sid,name):
    out=Path(out);path=out/(name+'_index.npz')
    if path.exists():
        meta=read(out/(name+'_index.json'))
        if sha256(path)!=meta['sha256'] or meta['cutoff']!=cutoff:raise ValueError('Changed saved index')
        return
    meta=read(out/'preparation.json');store=Store(meta['store']);stats=prefix_state(store,cutoff)
    write_json(out/f'prefix_{cutoff}.json',stats)
    old=read(Path(source)/'preparation.json');result=[]
    try:
        for s in store.series:
            sid=s['sid'];starts=complete_origins(s,starts_by_sid[sid])
            if not len(starts):continue
            with np.load(Path(source)/'banks'/f'{sid}.npz',allow_pickle=False) as bank:
                delta=np.diff(bank['starts']);stride=int(np.min(delta)) if len(delta) else 16
            retriever=PrefixRetriever(store,stats,sid,meta['period'],stride)
            namespace='foundation_test' if name=='test' else 'foundation'
            with np.load(out/namespace/f'{sid}.npz',allow_pickle=False) as f:
                ids=np.searchsorted(f['starts'],starts);np.testing.assert_array_equal(f['starts'][ids],starts);quantiles=f['quantiles'][ids]
            for p,q in zip(starts,quantiles):
                selected,random=retriever.select(store.window(sid,int(p),96),int(p),q)
                result.append((sid,int(p),selected,random))
            if sid%25==0:print('FORWARD INDEX',out.name,name,sid,'/',len(store.series),flush=True)
        np.savez_compressed(path,sid=np.array([r[0] for r in result]),start=np.array([r[1] for r in result]),
            selected_starts=np.stack([r[2] for r in result]),random_selected_starts=np.stack([r[3] for r in result]))
        write_json(out/(name+'_index.json'),dict(cutoff=cutoff,queries=len(result),sha256=sha256(path),stats_sha256=sha256(out/f'prefix_{cutoff}.json')))
    finally:store.close()


def materialize(source,out,name):
    source=Path(source);out=Path(out);meta=read(out/'preparation.json');index_meta=read(out/(name+'_index.json'));path=out/(name+'_index.npz')
    if sha256(path)!=index_meta['sha256']:raise ValueError('Changed case index')
    with np.load(path,allow_pickle=False) as f:d={k:f[k].copy() for k in f.files}
    n=len(d['sid']);d.update(raw_x=np.empty((n,96),np.float32),raw_y=np.empty((n,96),np.float32),
        raw_quantiles=np.empty((n,9,96),np.float32),probe_quantiles=np.empty((n,9,32),np.float32),std=np.empty(n,np.float32),shuffle_order=np.empty((n,10),np.int64))
    for prefix in ['', 'random_']:
        for key in ['case_past','case_future','case_baseline']:d[prefix+key]=np.empty((n,10,96),np.float32)
    store=Store(meta['store'])
    try:
        for sid in np.unique(d['sid']):
            sid=int(sid);ids=np.flatnonzero(d['sid']==sid);starts=d['start'][ids];raw=store.values(sid)
            namespace='foundation_test' if name=='test' else 'foundation'
            with np.load(out/namespace/f'{sid}.npz',allow_pickle=False) as f:
                loc=np.searchsorted(f['starts'],starts);np.testing.assert_array_equal(f['starts'][loc],starts)
                d['raw_quantiles'][ids]=f['quantiles'][loc];d['probe_quantiles'][ids]=f['probe'][loc]
            d['raw_x'][ids]=raw[starts[:,None]+np.arange(96)];d['raw_y'][ids]=raw[starts[:,None]+96+np.arange(96)]
            d['std'][ids]=max(store.series[sid]['memory_std'],1e-6)
            bankfile=source/'banks'/f'{sid}.npz'
            expected=read(source/'preparation.json')['banks'][str(sid)]
            if sha256(bankfile)!=expected:raise ValueError('Changed frozen reference bank')
            with np.load(bankfile,allow_pickle=False) as bank:
                for prefix in ['', 'random_']:
                    sel=d[prefix+'selected_starts'][ids];p=np.searchsorted(bank['starts'],sel)
                    np.testing.assert_array_equal(bank['starts'][p],sel)
                    d[prefix+'case_past'][ids]=bank['past'][p];d[prefix+'case_baseline'][ids]=bank['baseline'][p]
                    d[prefix+'case_future'][ids]=raw[sel[:,:,None]+96+np.arange(96)]
            for row in ids:
                rng=np.random.default_rng(20261015+sid*1000003+int(d['start'][row])+53);order=[]
                for j in [0,5]:
                    perm=rng.permutation(5)
                    while np.any(perm==np.arange(5)):perm=rng.permutation(5)
                    order.extend((j+perm).tolist())
                d['shuffle_order'][row]=order
        return d
    finally:store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--device',default='cuda');p.add_argument('--test',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    for name in a.datasets:foundation(Path(a.source)/name,Path(a.output)/name,a.device,a.test)
