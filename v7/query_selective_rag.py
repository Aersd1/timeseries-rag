"""Actual frozen Moirai + original 5+5 + chronological utility selector."""
from pathlib import Path
import numpy as np
import torch
from .common import sha256
from .prepare_unified_residual import read
from .benchmark_unified_residual import trace
from .benchmark_selective_rag import frozen_predictions,verified_source
from .query_unified_residual import UnifiedResidualPredictor
from .selective_rag import SETTINGS,UtilityNet,options,selector_features,score_selector,route


class SelectiveRagPredictor:
    """No current truth argument; rejected patches return foundation bit-for-bit."""
    def __init__(self,folder,device='cuda',adapter=None,arm='real'):
        if arm not in ['real','case_utility','value_only']:raise ValueError('Only explicitly specified deployable arms')
        self.folder=Path(folder);self.device=device;self.arm=arm
        run=read(self.folder/'run.json');contract=read(self.folder/'contract.json');selection=read(self.folder/'selection.json')
        if sha256(self.folder/'contract.json')!=run['contract_sha256']:raise ValueError('Changed selector contract')
        for file,key in [('policy.json','policy_sha256'),('selection.json','selection_sha256')]:
            if sha256(self.folder/file)!=contract[key]:raise ValueError('Changed selector configuration')
        source=Path(selection['source'])
        if verified_source(source)!=selection['source_hashes']:raise ValueError('Changed upstream forecasts')
        self.meta=read(source/'preparation.json');self.stats=read(source/'stats.json');self.history=trace(source,'validation')
        if sha256(source/'validation.npz')!=selection['source_data_sha256']:raise ValueError('Changed competence history')
        hp=frozen_predictions(source,'validation');self.history_options=options(self.history,self.stats,self.meta['period'],arm,hp.get(arm))
        self.minimum_start=max(selection['test_start'],selection['split']['confirmation']['last_label_end'])
        self.policy=read(self.folder/'policy.json')[arm];entry=next(m for m in selection['models'] if m['arm']==arm)
        path=self.folder/entry['file']
        if sha256(path)!=entry['sha256']:raise ValueError('Changed selector model')
        saved=torch.load(path,map_location='cpu',weights_only=True);self.normal=saved['normal'];self.models=[]
        for state in saved['models']:
            model=UtilityNet(self.normal['input_width'],SETTINGS['width']);model.load_state_dict(state);self.models.append(model.to(device).eval())
        self.previous=UnifiedResidualPredictor(source,device,adapter)

    def close(self):self.previous.close()

    def predict(self,history,sid,start):
        x=np.asarray(history,dtype=np.float32);adapter=self.previous.previous.previous.original.adapter
        if x.shape!=(96,) or not np.isfinite(x).all():raise ValueError('Expected observed finite 96')
        q=adapter.forecast(x[None],96).cpu().numpy();probe=adapter.forecast(x[None,:64],32).cpu().numpy()
        return self.predict_cached(x,sid,start,q,probe)

    def predict_cached(self,history,sid,start,q,probe):
        sid=int(sid);start=int(start);x=np.asarray(history,dtype=np.float32)
        if start<self.minimum_start:raise ValueError('Selector was trained/confirmed after this query')
        if x.shape!=(96,) or not np.isfinite(x).all():raise ValueError('Expected observed finite 96')
        old=self.previous.predict_cached(x,sid,start,q,probe);selected=np.asarray(old['selected_starts'],dtype=np.int64);bank=self.previous.bank(sid)
        if np.any(selected+192>min(start,int(bank['end']))):raise ValueError('Unobserved or overlapping case')
        ids=np.searchsorted(bank['starts'],selected);np.testing.assert_array_equal(bank['starts'][ids],selected)
        raw=self.previous.previous.previous.original.synchronous_values[sid]
        data=dict(raw_x=x[None],raw_quantiles=np.asarray(q),probe_quantiles=np.asarray(probe),sid=np.array([sid]),start=np.array([start]),
            selected_starts=selected[None],case_past=bank['past'][ids][None],case_baseline=bank['baseline'][ids][None],
            case_future=raw[selected[:,None]+96+np.arange(96)][None])
        frozen=np.asarray(old['forecasts']['raw_real'],dtype=np.float32)[None] if self.arm=='real' else None
        candidates=options(data,self.stats,self.meta['period'],self.arm,frozen)
        xf=selector_features(data,candidates,self.stats,self.meta['period'],self.arm,self.history,self.history_options)
        scores=score_selector(self.models,self.normal,xf,self.device);p=self.policy
        pred,mask,choice=route(np.asarray(q)[:,4],candidates,scores,p['threshold'],p['strength'],p['enabled'])
        return dict(forecast=pred[0].tolist(),moirai=np.asarray(q)[0,4].tolist(),primary=self.arm,
            selected_starts=selected.tolist(),adopted_patches=mask[0].tolist(),scores=scores[0].tolist(),
            case_indices=choice[0].tolist() if self.arm=='case_utility' else None,
            retrieved_cases=10,used_cases=int(len(np.unique(choice[mask]))) if self.arm=='case_utility' else (10 if mask.any() else 0),
            current_future_used=False,fallback='direct frozen Moirai',policy_enabled=p['enabled'])
