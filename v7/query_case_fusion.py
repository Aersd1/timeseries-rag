"""Deploy frozen learned fusion from current96 history and observed five or ten cases."""
import json
from pathlib import Path
import numpy as np
import torch
from .common import sha256
from .query_dual_five import DualFivePredictor
from .query_guarded_bank import GuardedBankPredictor
from .aligned_case_fusion import CaseFusion,MODES,inputs,tensor_inputs,forecast
from .benchmark_case_fusion import apply


class LearnedCasePredictor:
    def __init__(self,store_path,folder,device='cuda',adapter=None):
        self.folder=Path(folder);self.device=device
        self.run=json.loads((self.folder/'run.json').read_text());self.contract=json.loads((self.folder/'contract.json').read_text())
        if sha256(self.folder/'contract.json')!=self.run['contract_sha256']:raise ValueError('Changed fusion contract')
        for filename,key in [('preparation.json','preparation_sha256'),('stats.json','stats_sha256'),
            ('selection.json','selection_sha256'),('policy.json','policy_sha256')]:
            if sha256(self.folder/filename)!=self.contract[key]:raise ValueError('Changed fusion training/policy metadata')
        meta=json.loads((self.folder/'preparation.json').read_text());source=Path(meta['source'])
        if sha256(source/'run.json')!=meta['source_run_sha256'] or sha256(source/'contract.json')!=meta['source_contract_sha256']:
            raise ValueError('Changed frozen retrieval source')
        self.case_count=meta.get('case_count',10)
        if self.case_count==5:
            source_contract=json.loads((source/'contract.json').read_text())
            self.previous=GuardedBankPredictor(store_path,source_contract['source'],device,adapter)
            p=self.previous.original
        else:
            self.previous=DualFivePredictor(store_path,source,device,adapter)
            p=self.previous.previous.original
        self.original=p;self.limits=meta['limits']
        if p.store.meta['data_id']!=meta['source_data_id']:raise ValueError('Wrong dataset store')
        self.stats=json.loads((self.folder/'stats.json').read_text());self.stats['memory_std']=p.stds.tolist()
        self.period=meta['period'];self.policies=json.loads((self.folder/'policy.json').read_text())
        selection=json.loads((self.folder/'selection.json').read_text());self.models={mode:[] for mode in MODES}
        for record in selection['models']:
            path=self.folder/record['file']
            if sha256(path)!=record['sha256']:raise ValueError('Changed learned fusion weights')
            state=torch.load(path,map_location='cpu',weights_only=True)
            model=CaseFusion(state['variables'],state['width'],state.get('conditioned',False),state.get('paired',False),state.get('metadata_width',5));model.load_state_dict(state['state_dict']);model.to(device).eval()
            self.models[record['mode']].append(model)

    def close(self):self.previous.close()

    def predict(self,history,sid,start):
        x=np.asarray(history,dtype=np.float32)
        if x.shape!=(96,) or not np.isfinite(x).all():raise ValueError('Expected finite96 history')
        adapter=self.original.adapter
        q=adapter.forecast(x[None],96).cpu().numpy();probe=adapter.forecast(x[None,:64],32).cpu().numpy()
        return self.predict_cached(x,sid,start,q,probe)

    def predict_cached(self,history,sid,start,q,probe):
        x=np.asarray(history,dtype=np.float32);sid=int(sid);start=int(start)
        previous=self.previous.predict_cached(x,sid,start,q,probe)
        selected=previous['selected_starts'];p=self.original;limit=int(self.limits[str(sid)])
        if np.any(np.asarray(selected)+192>min(start,limit)):raise ValueError('Unobserved or overlapping case')
        values=dict(raw_x=x[None],raw_quantiles=np.asarray(q),sid=np.array([sid]),start=np.array([start]),
            selected_starts=np.asarray(selected)[None],
            case_past=np.stack([p.synchronous_values[sid][t:t+96] for t in selected])[None],
            case_future=np.stack([p.synchronous_values[sid][t+96:t+192] for t in selected])[None])
        if self.stats.get('residual_reference'):
            values['reference']=np.asarray(previous['forecasts']['guarded_rag'])[None]
            values['probe_quantiles']=np.asarray(probe)
        forecasts={'moirai_direct':np.asarray(q)[0,4].tolist()}
        for mode,models in self.models.items():
            features,_,scale=inputs(values,self.stats,self.period,mode);features=tensor_inputs(features,self.device)
            raw=np.mean([forecast(model,features,values,scale) for model in models],axis=0).astype(np.float32)
            forecasts['raw_'+mode]=raw[0].tolist();forecasts['calibrated_'+mode]=apply(self.policies[mode],values,raw)[0].tolist()
        if self.case_count==5 and not self.stats.get('residual_reference'):
            incumbent=np.asarray(previous['forecasts']['guarded_rag'])[None]
            for mode in MODES:
                proposal=incumbent+np.asarray(forecasts['raw_'+mode])[None]-np.asarray(q)[:,4]
                forecasts['raw_increment_'+mode]=proposal[0].tolist()
                forecasts['increment_'+mode]=apply(self.policies['increment_'+mode],values,proposal,incumbent)[0].tolist()
            proposal=incumbent+np.asarray(forecasts['raw_wiener'])[None]-np.asarray(forecasts['raw_without_retrieval'])[None]
            forecasts['increment_difference']=apply(self.policies['increment_difference'],values,proposal,incumbent)[0].tolist()
        primary='increment_wiener' if self.case_count==5 and not self.stats.get('residual_reference') else 'wiener'
        uses_cases=self.policies[primary][str(sid)]['beta']>0
        if self.case_count==5 and not uses_cases:
            old_policy=self.previous.policy[str(sid)]
            if old_policy['branch']=='incumbent':old_policy=p.policies['routed'][str(sid)]
            uses_cases=old_policy['beta']>0 and old_policy['family'] not in ['direct','encoder_only','direct_encoder']
        return dict(forecasts=forecasts,selected_starts=selected,length=96,horizon=96,retrieved_cases=self.case_count,
            primary=primary,primary_forecast=primary if primary.startswith('increment_') else 'calibrated_'+primary,new_fusion_applied=self.policies[primary][str(sid)]['beta']>0,used_cases=self.case_count if uses_cases else 0,
            foundation_frozen=True,query_future_used=False,training='External adapter only; three fixed seeds')
