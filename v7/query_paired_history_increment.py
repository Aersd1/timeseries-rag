"""History-only serving for the paired-history increment experiment."""
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from .common import sha256
from .dense_forward import read,PrefixRetriever
from .query_forward_selectors import ObservedStore
from .moirai import MoiraiAdapter
from .benchmark_forward_selectors import load_models
from .benchmark_unified_residual import predict as base_predict
from .forward_selectors import prediction,apply_policy
from .paired_history_increment import pack,with_base,predict
from .benchmark_paired_history_increment import (baseline_model,load_increment,final_policy,
    utility_inputs,carrier,implementation)


class PairedHistoryPredictor:
    def __init__(self,folder,device='cuda',adapter=None):
        self.folder=Path(folder);self.device=device;contract=read(self.folder/'contract.json');protocol=read(self.folder/'protocol.json')
        if contract['code']!=implementation() or sha256(self.folder/'protocol.json')!=contract['protocol_sha256']:raise ValueError('Changed implementation')
        if sha256(self.folder/'policy.json')!=contract['policy_sha256']:raise ValueError('Changed policy')
        self.forward=Path(protocol['forward']);self.meta=read(self.forward/'preparation.json')
        if sha256(self.forward/'selector_contract.json')!=protocol['upstream_contract_sha256']:raise ValueError('Changed retrieval source')
        self.source=Path(self.meta['source']);self.store=Store(self.meta['store'])
        cuts=self.meta['layout']['cuts'];self.stats=read(self.forward/f'prefix_{cuts[-1]}.json');self.anchor=read(self.forward/f'prefix_{cuts[0]}.json')
        self.base_model,_=baseline_model(self.forward,'final',device)
        manifest=read(self.folder/'final_experts.json');entry=manifest['models']['paired']
        if sha256(self.folder/'final_experts.json')!=contract['phases']['final'] or sha256(self.folder/entry['file'])!=entry['sha256']:raise ValueError('Changed paired model')
        self.model=load_increment(self.folder/entry['file'],device)
        self.mode,self.policy,self.base_ok=final_policy(self.folder,'paired');key=f'paired__{self.mode}'
        path=self.folder/'utilities'/key
        if sha256(path.with_suffix('.json'))!=contract['models'][key]:raise ValueError('Changed utility metadata')
        self.nets,self.normal=load_models(path,device)
        self.adapter=adapter if adapter is not None else MoiraiAdapter.pretrained(self.meta['moirai'],device)
        self.retrievers={};self.banks={}

    def close(self):self.store.close()

    def predict(self,history,sid,start):
        x=np.asarray(history,np.float32)
        if x.shape!=(96,) or not np.isfinite(x).all():raise ValueError('Expected finite96 history')
        q=self.adapter.forecast(x[None],96).cpu().numpy();probe=self.adapter.forecast(x[None,:64],32).cpu().numpy()
        return self.predict_cached(x,sid,start,q,probe)

    def predict_cached(self,history,sid,start,q,probe):
        x=np.asarray(history,np.float32);sid=int(sid);start=int(start);q=np.asarray(q,np.float32);probe=np.asarray(probe,np.float32)
        if x.shape!=(96,) or q.shape!=(1,9,96) or probe.shape!=(1,9,32):raise ValueError('Expected96 history, nine96 quantiles and nine32 probes')
        if not all(np.isfinite(v).all() for v in [x,q,probe]):raise ValueError('Nonfinite query')
        if start<self.meta['layout']['validation_end']:raise ValueError('Query before policy confirmation')
        observed=ObservedStore(self.store,start+96)
        if sid not in self.banks:
            path=self.source/'banks'/f'{sid}.npz'
            if sha256(path)!=read(self.source/'preparation.json')['banks'][str(sid)]:raise ValueError('Changed old reference bank')
            with np.load(path,allow_pickle=False) as f:self.banks[sid]={k:f[k].copy() for k in f.files}
            stride=int(np.diff(self.banks[sid]['starts']).min())
            self.retrievers[sid]=PrefixRetriever(observed,self.stats,sid,self.meta['period'],stride)
        retriever=self.retrievers[sid];retriever.store=observed
        selected,random=retriever.select(x,start,q[0]);bank=self.banks[sid];raw=observed.values(sid)
        d=dict(raw_x=x[None],raw_quantiles=q,probe_quantiles=probe,sid=np.array([sid]),start=np.array([start]),
            std=np.array([self.store.series[sid]['memory_std']]))
        for prefix,sel in [('',selected),('random_',random)]:
            if np.any(sel+192>min(start,retriever.raw.end)):raise ValueError('Self/overlapping or unavailable case')
            loc=np.searchsorted(bank['starts'],sel);np.testing.assert_array_equal(bank['starts'][loc],sel)
            d[prefix+'selected_starts']=sel[None];d[prefix+'case_past']=bank['past'][loc][None]
            d[prefix+'case_baseline']=bank['baseline'][loc][None];d[prefix+'case_future']=raw[sel[:,None]+96+np.arange(96)][None]
        # Actual histories must be available and match the retrieval-cache values.
        np.testing.assert_array_equal(d['case_past'][0],raw[selected[:,None]+np.arange(96)])
        rng=np.random.default_rng(20261015+sid*1000003+start+53);order=[]
        for offset in [0,5]:
            perm=rng.permutation(5)
            while np.any(perm==np.arange(5)):perm=rng.permutation(5)
            order.extend((perm+offset).tolist())
        d['shuffle_order']=np.array(order)[None]
        raw_base=q[:,4] if self.mode==0 else base_predict(self.base_model,d,self.stats,self.meta['period'],'query_only',self.device)
        base=raw_base if self.base_ok else q[:,4]
        p,scale=pack(d,self.stats,self.meta['period']);p=with_base(p,raw_base,d,scale,self.mode)
        candidate=predict(self.model,p,raw_base,scale,self.device)
        xx=utility_inputs(d,raw_base,candidate,self.anchor,self.meta['period'],'paired',self.mode)
        u=prediction('summary_mlp',self.nets,self.normal,xx,self.device)
        output,mask=apply_policy(carrier(d,base),candidate,u,self.policy)
        return dict(forecast=output[0].tolist(),candidate_forecast=candidate[0].tolist(),baseline=base[0].tolist(),
            moirai=q[0,4].tolist(),selected_starts=selected.tolist(),adopted_patches=mask[0].tolist(),
            baseline_mode=self.mode,case_history_points=96,case_continuation_points=96,case_inputs=10,
            history_in_value_path=True,case_branch_adopted=bool(mask.any()),query_future_used=False)
