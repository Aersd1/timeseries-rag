"""Serve the validation-frozen portfolio with a physically history-limited store."""
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from .common import sha256
from .moirai import MoiraiAdapter
from .dense_forward import read,PrefixRetriever
from .forward_experts import model_from
from .benchmark_unified_residual import predict as expert_predict
from .benchmark_forward_selectors import load_models
from .forward_selectors import inputs,prediction,apply_policy


class ObservedStore:
    def __init__(self,store,end):self.original=store;self.series=store.series;self.end=int(end)
    def values(self,sid):return self.original.values(sid)[:self.end]
    def window(self,sid,start,length):
        if start<0 or start+length>self.end:raise ValueError('Attempted to read unobserved query future')
        return self.original.window(sid,start,length)


class ForwardSelectorPredictor:
    def __init__(self,folder,device='cuda',adapter=None,arm='real',cohort='forward_full',family='portfolio'):
        self.folder=Path(folder);self.device=device;self.arm=arm;self.cohort=cohort;self.family=family
        self.meta=read(self.folder/'preparation.json');self.contract=read(self.folder/'selector_contract.json');run=read(self.folder/'run.json')
        if sha256(self.folder/'selector_contract.json')!=run['contract_sha256']:raise ValueError('Changed selector contract')
        if sha256(self.folder/'selector_policy.json')!=self.contract['policy_sha256']:raise ValueError('Changed policy')
        self.key=cohort+'__'+arm+'__'+family;self.policy=read(self.folder/'selector_policy.json')[self.key]
        self.selected_family=self.policy.get('selected_family',family)
        self.store=Store(self.meta['store']);self.source=Path(self.meta['source']);cuts=self.meta['layout']['cuts']
        self.stats=read(self.folder/f'prefix_{cuts[-1]}.json');self.anchor=read(self.folder/f'prefix_{cuts[0]}.json')
        r=read(self.folder/'final_experts.json');entry=next(m for m in r['models'] if m['mode']==arm);path=self.folder/entry['file']
        if sha256(self.folder/f'prefix_{cuts[-1]}.json')!=r['stats_sha256'] or sha256(self.folder/f'prefix_{cuts[0]}.json')!=self.contract['provenance']['anchor_sha256']:raise ValueError('Changed forecast/selector normalization')
        if sha256(self.folder/'foundation_manifest.json')!=self.contract['provenance']['foundation_manifest_sha256']:raise ValueError('Changed historical forecasts')
        if sha256(path)!=entry['sha256']:raise ValueError('Changed forecast expert')
        self.expert=model_from(torch.load(path,map_location='cpu',weights_only=True),device)
        self.models=None
        if self.selected_family!='static':
            path=self.folder/'selectors'/(cohort+'__'+arm+'__'+self.selected_family)
            if sha256(path.with_suffix('.json'))!=self.contract['models'][path.name]:raise ValueError('Changed selector manifest')
            self.models,self.normal=load_models(path,device)
        self.adapter=adapter if adapter is not None else MoiraiAdapter.pretrained(self.meta['moirai'],device)
        self.history={};self.retrievers={};self.banks={}

    def close(self):self.store.close()

    def history_for(self,sid):
        if sid in self.history:return self.history[sid]
        starts=[];forecast=[]
        for phase in ['fold0','fold1','fold2','final']:
            with np.load(self.folder/(phase+'_predictions.npz'),allow_pickle=False) as f:
                rows=np.flatnonzero(f['sid']==sid);starts.extend(f['start'][rows].tolist());forecast.extend(f[self.arm][rows])
        starts=np.asarray(starts,dtype=np.int64);raw=self.store.values(sid)
        if np.any(starts+192>self.meta['layout']['validation_end']):raise ValueError('Unobserved competence label')
        if sha256(self.folder/'foundation'/f'{sid}.npz')!=read(self.folder/'foundation_manifest.json')['files'][str(sid)]:raise ValueError('Changed competence forecast')
        with np.load(self.folder/'foundation'/f'{sid}.npz',allow_pickle=False) as f:
            ids=np.searchsorted(f['starts'],starts);np.testing.assert_array_equal(f['starts'][ids],starts);base=f['quantiles'][ids,4]
        h=dict(sid=np.full(len(starts),sid),start=starts,raw_y=np.asarray(raw[starts[:,None]+96+np.arange(96)]),
            raw_quantiles=np.broadcast_to(base[:,None],(len(base),9,96)))
        self.history[sid]=(h,np.asarray(forecast,dtype=np.float32));return self.history[sid]

    def predict(self,history,sid,start):
        x=np.asarray(history,dtype=np.float32)
        if x.shape!=(96,) or not np.isfinite(x).all():raise ValueError('Expected finite96 history')
        q=self.adapter.forecast(x[None],96).cpu().numpy();probe=self.adapter.forecast(x[None,:64],32).cpu().numpy()
        return self.predict_cached(x,sid,start,q,probe)

    def predict_cached(self,history,sid,start,q,probe):
        sid=int(sid);start=int(start);x=np.asarray(history,dtype=np.float32)
        if x.shape!=(96,) or not np.isfinite(x).all():raise ValueError('Expected finite96 history')
        if start<self.meta['layout']['validation_end']:raise ValueError('Selector was confirmed after this origin')
        observed=ObservedStore(self.store,start+96)
        if sid not in self.retrievers:
            bankpath=self.source/'banks'/f'{sid}.npz';expected=read(self.source/'preparation.json')['banks'][str(sid)]
            if sha256(bankpath)!=expected:raise ValueError('Changed frozen Moirai reference bank')
            with np.load(bankpath,allow_pickle=False) as f:
                bank={k:f[k].copy() for k in f.files}
            self.banks[sid]=bank;stride=int(np.diff(bank['starts']).min())
            self.retrievers[sid]=PrefixRetriever(observed,self.stats,sid,self.meta['period'],stride)
        retriever=self.retrievers[sid];retriever.store=observed
        selected,random=retriever.select(x,start,np.asarray(q)[0]);bank=self.banks[sid];raw=observed.values(sid)
        d=dict(raw_x=x[None],raw_quantiles=np.asarray(q),probe_quantiles=np.asarray(probe),sid=np.array([sid]),start=np.array([start]),std=np.array([self.store.series[sid]['memory_std']]))
        for prefix,sel in [('',selected),('random_',random)]:
            if np.any(sel+192>min(start,retriever.raw.end)):raise ValueError('Unknown/self-overlapping episode')
            idx=np.searchsorted(bank['starts'],sel);np.testing.assert_array_equal(bank['starts'][idx],sel)
            d[prefix+'selected_starts']=sel[None];d[prefix+'case_past']=bank['past'][idx][None]
            d[prefix+'case_baseline']=bank['baseline'][idx][None];d[prefix+'case_future']=raw[sel[:,None]+96+np.arange(96)][None]
        rng=np.random.default_rng(20261015+sid*1000003+start+53);order=[]
        for j in [0,5]:
            perm=rng.permutation(5)
            while np.any(perm==np.arange(5)):perm=rng.permutation(5)
            order.extend((j+perm).tolist())
        d['shuffle_order']=np.array(order)[None]
        candidate=expert_predict(self.expert,d,self.stats,self.meta['period'],self.arm,self.device)
        util=None
        if self.selected_family!='static':
            h,hp=self.history_for(sid);xx=inputs(d,candidate,self.anchor,self.meta['period'],self.arm,h,hp,raw=self.selected_family=='raw_sequence')
            util=prediction(self.selected_family,self.models,self.normal,xx,self.device)
        forecast,mask=apply_policy(d,candidate,util,self.policy)
        return dict(forecast=forecast[0].tolist(),candidate_forecast=candidate[0].tolist(),
            moirai=np.asarray(q)[0,4].tolist(),selected_starts=selected.tolist(),
            adopted_patches=mask[0].tolist(),policy_enabled=self.policy['enabled'],family=self.selected_family,
            retrieved_cases=10,case_inputs=0 if self.arm=='query_only' else 10,
            case_branch_adopted=bool(mask.any() and self.arm!='query_only'),query_future_used=False)
