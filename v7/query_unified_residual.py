"""Actual 5+5 retrieval and independent frozen ablation networks."""
from pathlib import Path
import numpy as np
import torch
from .common import sha256
from .prepare_unified_residual import read
from .query_dual_five import DualFivePredictor
from .unified_residual import MODES,UnifiedResidualFusion,matched_starts
from .benchmark_unified_residual import predict,apply_confirmed,fixed_transfer


class UnifiedResidualPredictor:
    def __init__(self,folder,device='cuda',adapter=None):
        self.folder=Path(folder);self.device=device;self.meta=read(self.folder/'preparation.json')
        self.run=read(self.folder/'run.json');contract=read(self.folder/'contract.json')
        if sha256(self.folder/'contract.json')!=self.run['contract_sha256']:raise ValueError('Changed contract')
        for file,key in [('preparation.json','preparation_sha256'),('stats.json','stats_sha256'),('policy.json','policy_sha256'),('selection.json','selection_sha256')]:
            if sha256(self.folder/file)!=contract[key]:raise ValueError('Changed deployment inputs')
        source=Path(self.meta['source'])
        if sha256(source/'preparation.json')!=self.meta['source_preparation_sha256'] or sha256(source/'run.json')!=self.meta['source_run_sha256']:raise ValueError('Changed frozen source')
        original=read(source/'preparation.json');self.previous=DualFivePredictor(self.meta['store'],original['source'],device,adapter)
        if self.previous.previous.original.store.meta['data_id']!=self.meta['source_data_id']:raise ValueError('Wrong source data')
        self.stats=read(self.folder/'stats.json');self.policies=read(self.folder/'policy.json');self.banks={};self.models={k:[] for k in MODES}
        self.selection=read(self.folder/'selection.json')
        for r in self.selection['models']:
            path=self.folder/r['file']
            if sha256(path)!=r['sha256']:raise ValueError('Changed neural parameters')
            saved=torch.load(path,map_location='cpu',weights_only=True);model=UnifiedResidualFusion(saved['variables'],saved['width'])
            model.load_state_dict(saved['state_dict']);self.models[r['mode']].append(model.to(device).eval())

    def close(self):self.previous.close()

    def bank(self,sid):
        if sid not in self.banks:
            path=self.folder/'banks'/f'{sid}.npz'
            if sha256(path)!=self.meta['banks'][str(sid)]:raise ValueError('Changed reference bank')
            with np.load(path,allow_pickle=False) as f:self.banks[sid]={k:f[k].copy() for k in f.files}
            if len(self.banks)>4:self.banks.pop(next(iter(self.banks)))
        return self.banks[sid]

    def predict(self,history,sid,start):
        x=np.asarray(history,dtype=np.float32);adapter=self.previous.previous.original.adapter
        if x.shape!=(96,) or not np.isfinite(x).all():raise ValueError('Expected finite observed96')
        q=adapter.forecast(x[None],96).cpu().numpy();probe=adapter.forecast(x[None,:64],32).cpu().numpy()
        return self.predict_cached(x,sid,start,q,probe)

    def predict_cached(self,history,sid,start,q,probe):
        x=np.asarray(history,dtype=np.float32);sid=int(sid);start=int(start)
        if x.shape!=(96,) or not np.isfinite(x).all():raise ValueError('Expected finite observed96')
        old=self.previous.predict_cached(x,sid,start,q,probe);selected=np.array(old['selected_starts'],dtype=np.int64)
        bank=self.bank(sid)
        if np.any(selected+192>min(start,int(bank['end']))):raise ValueError('Overlapping/self/unobserved episode')
        ids=np.searchsorted(bank['starts'],selected);np.testing.assert_array_equal(bank['starts'][ids],selected)
        p=self.previous.previous.original;raw=p.synchronous_values[sid]
        seed=20261015+sid*1000003+start
        random=matched_starts(bank['starts'],bank['past'],selected,start,self.meta['period'],self.stats['memory_mean'][sid],self.stats['memory_std'][sid],seed)
        ri=np.searchsorted(bank['starts'],random);rng=np.random.default_rng(seed+17);order=[]
        for j in [0,5]:
            perm=rng.permutation(5)
            if np.array_equal(perm,np.arange(5)):perm=np.roll(perm,1)
            order.extend((j+perm).tolist())
        data=dict(raw_x=x[None],raw_quantiles=np.asarray(q),probe_quantiles=np.asarray(probe),sid=np.array([sid]),start=np.array([start]),
            selected_starts=selected[None],case_past=bank['past'][ids][None],case_baseline=bank['baseline'][ids][None],
            case_future=raw[selected[:,None]+96+np.arange(96)][None],random_selected_starts=random[None],
            random_case_past=bank['past'][ri][None],random_case_baseline=bank['baseline'][ri][None],
            random_case_future=raw[random[:,None]+96+np.arange(96)][None],shuffle_order=np.array(order)[None])
        outputs={'moirai_direct':np.asarray(q)[:,4]}
        for mode,models in self.models.items():outputs['raw_'+mode]=np.mean([predict(m,data,self.stats,self.meta['period'],mode,self.device) for m in models],0).astype(np.float32)
        qbase=apply_confirmed(data,outputs['raw_query_only'],outputs['moirai_direct'],self.policies['query_only']);outputs['query_confirmed']=qbase
        for mode in MODES[1:]:
            outputs['own_'+mode]=apply_confirmed(data,outputs['raw_'+mode],outputs['moirai_direct'],self.policies['own_'+mode])
            outputs['extra_'+mode]=apply_confirmed(data,outputs['raw_'+mode],qbase,self.policies['extra_'+mode])
        outputs['fixed_transfer']=fixed_transfer(data,self.stats,self.meta['period'])
        outputs['confirmed_transfer']=apply_confirmed(data,outputs['fixed_transfer'],outputs['moirai_direct'],self.policies['transfer_only'])
        adopted=self.policies['extra_real']['aggregate_guard']['accepted'] and self.policies['extra_real']['policy'][str(sid)]['beta']>0
        return dict(forecasts={k:v[0].tolist() for k,v in outputs.items()},selected_starts=selected.tolist(),random_starts=random.tolist(),
            primary='extra_real',additional_rag_adopted=adopted,retrieved_cases=10,used_cases=10 if adopted else 0,
            foundation_frozen=True,query_future_used=False)
