"""Persistent history-only prediction with a learned joint future-state router."""
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import sha256
from .moirai import MoiraiAdapter
from .trend_rag import ResidualPool
from .error_correction import retrieve
from .query_past_error_adapter import scale_data
from .past_error_adapter import feature
from .benefit_rerank import covariance
from .safe_error_policy import bound_delta,apply_policy
from .future_state_encoder import history_features,prototype_state,predict,decode
from .benchmark_future_state_rag import retrieve_forecast
from .complementary_rag import candidate_innovations


class FutureStatePredictor:
    def __init__(self,store_path,folder,device='cuda',adapter=None):
        self.folder=Path(folder);self.run=json.loads((self.folder/'run.json').read_text(encoding='utf-8'))
        self.checkpoint=Path(self.run['source_checkpoint']);self.base_run=json.loads((self.checkpoint/'run.json').read_text(encoding='utf-8'))
        self.cache=Path(self.base_run['source_cache']);self.meta=json.loads((self.cache/'manifest.json').read_text(encoding='utf-8'))
        if sha256(self.cache/'manifest.json')!=self.base_run['source_manifest_sha256']:raise ValueError('Changed source data cache')
        self.grouped=self.run.get('grouped',False)
        for name,key in [('policy.json','policy_sha256'),('selection.json','selection_sha256')]:
            if sha256(self.folder/name)!=self.run[key]:raise ValueError('Changed future encoder or routing policy')
        for name,key in [('context_plan.json','context_plan_sha256'),('policy.json','policy_sha256')]:
            if sha256(self.checkpoint/name)!=self.base_run[key]:raise ValueError('Changed base expert metadata')
        if self.grouped:
            if sha256(self.folder/'groups.json')!=self.run['groups_sha256']:raise ValueError('Changed channel groups')
            self.groups=json.loads((self.folder/'groups.json').read_text(encoding='utf-8'));self.encoders={}
        else:
            if sha256(self.folder/'future_encoder.npz')!=self.run['encoder_sha256']:raise ValueError('Changed encoder')
            with np.load(self.folder/'future_encoder.npz',allow_pickle=False) as f:self.encoder={k:f[k].copy() for k in f.files}
        self.policies=json.loads((self.folder/'policy.json').read_text(encoding='utf-8'))
        self.base_policies=json.loads((self.checkpoint/'policy.json').read_text(encoding='utf-8'))['without_rag']
        self.selection=json.loads((self.folder/'selection.json').read_text(encoding='utf-8'))
        self.plan=json.loads((self.checkpoint/'context_plan.json').read_text(encoding='utf-8'))
        self.store=Store(store_path)
        if self.store.meta['data_id']!=self.base_run['source_data_id']:raise ValueError('Wrong synchronous store')
        self.means=np.array([s['memory_mean'] for s in self.store.series]);self.stds=np.array([max(s['memory_std'],1e-6) for s in self.store.series])
        self.synchronous_values=[np.array(self.store.values(s['sid']),dtype=np.float32) for s in self.store.series]
        self.adapter=adapter if adapter is not None else MoiraiAdapter.pretrained(self.meta['moirai'],device);self.pools={};self.base_states={}

    def close(self):self.store.close()

    def joint_window(self,start,length,channels=None):
        channels=list(range(len(self.store.series))) if channels is None else channels
        x=np.stack([self.synchronous_values[j][start:start+length] for j in channels],axis=1)
        if x.shape!=(length,len(channels)):raise ValueError('Incomplete synchronous window')
        return x

    def predict(self,history,sid,start):
        x=np.asarray(history,dtype=np.float32)
        if x.shape!=(96,) or not np.isfinite(x).all():raise ValueError('Expected finite96-point history')
        q=self.adapter.forecast(x[None],96).cpu().numpy();probe=self.adapter.forecast(x[None,:64],32).cpu().numpy()
        return self.predict_cached(x,sid,start,q,probe)

    def predict_cached(self,history,sid,start,q,probe,metadata=None,return_components=False,validation=False):
        """Replay fixed foundation outputs; validation is for calibration only."""
        x=np.asarray(history,dtype=np.float32);s=self.store.series[sid]
        if x.shape!=(96,) or not np.isfinite(x).all():raise ValueError('Expected finite96-point history')
        if start<(s['train_end'] if validation else s['validation_end']):raise ValueError('History outside calibration/deployment split')
        grouped=getattr(self,'grouped',False)
        group=self.groups[str(sid)] if grouped else list(range(len(self.store.series)))
        local_sid=group.index(sid)
        if grouped:
            if sid not in self.encoders:
                name=f'encoder_{sid}.npz'
                if sha256(self.folder/name)!=self.run['group_model_hashes'][name]:raise ValueError('Changed group encoder')
                with np.load(self.folder/name,allow_pickle=False) as f:self.encoders[sid]={k:f[k].copy() for k in f.files}
            encoder=self.encoders[sid]
        else:encoder=self.encoder
        q=np.asarray(q);probe=np.asarray(probe)
        if q.shape!=(1,9,96) or probe.shape!=(1,9,32):raise ValueError('Unexpected foundation output shape')
        known_error=x[None,64:]-probe[:,4]
        if sid not in self.pools:
            self.pools[sid]=ResidualPool(self.store,sid,self.meta['window'],stride=self.meta.get('memory_stride',1))
            name=f'past_{sid}_without_rag.npz'
            if sha256(self.checkpoint/name)!=self.base_run['past_hashes'][name]:raise ValueError('Changed base expert weights')
            with np.load(self.checkpoint/name,allow_pickle=False) as f:self.base_states[sid]={k:f[k].copy() for k in f.files}
        if metadata is None:_,m=retrieve(x,q[0],self.pools[sid],start,self.meta['window'])
        else:m=metadata
        std=np.float32(self.stds[sid]);data=dict(raw_x=x[None],raw_quantiles=q,base=q[:,4],std=np.array([std]),metric_std=np.array([std]),
            sid=np.array([sid]),start=np.array([start]),errors=np.zeros((1,5,96)),**{k:m[k][None] for k in ['weights','selected_starts','candidate_starts','history_scores','future_scores']})
        if self.base_run.get('annual_calendar'):data['annual_calendar']=True
        if self.base_run.get('adaptive_scale'):scale_data(data)
        f=feature(self.store,data,[0],s,self.base_run['period'],self.plan,known_error,False)
        p=self.base_policies[str(sid)];beta=p.get('proposed_beta',0.)
        base=q[:,4]+beta*bound_delta(covariance(self.base_states[sid],f)*data['std'][:,None],q)
        joint=self.joint_window(start,96,group);joint[:,local_sid]=x
        width=int(encoder.get('block_width',16));history_width=int(encoder.get('history_width',16))
        code=predict(encoder,history_features(joint[None],np.array([start]),self.means[group],self.stds[group],self.base_run['period'] or 1,weekly=grouped,block_width=history_width))[0]
        candidate_starts=m['candidate_starts']
        if np.any(candidate_starts+192>min(start,s['memory_end'])):raise ValueError('Self or overlapping future prototype')
        episodes=np.stack([self.joint_window(int(t),192,group) for t in candidate_starts]);past=episodes[:,:96];future=episodes[:,96:]
        prototypes=prototype_state(past,future,self.means[group],self.stds[group],width)
        order=np.argsort(candidate_starts);library=candidate_starts[order]
        prototypes={k:v[order] for k,v in prototypes.items()};past=past[order];future=future[order]
        weight=self.selection['groups'][str(sid)]['retrieval_weight'] if grouped else self.selection['retrieval_weight']
        rag,selected,weights=retrieve_forecast(data,0,base[0],code,encoder['precision'],None,prototypes,past,future,library,weight,channel_index=local_sid)
        scale=max(float(x.std()),.1*self.stds[sid]);level=float(x[-16:].mean())
        encoded_path=level+decode(code,len(group),width)[:,local_sid]*scale
        deltas={kind:bound_delta((v*std)[None],q)/std for kind,v in [('encoder_only',(encoded_path-base[0])/std),('future_state_rag',rag)]}
        for fraction in [.25,.5,.75]:deltas[f'mix_{fraction}']=(1-fraction)*deltas['encoder_only']+fraction*deltas['future_state_rag']
        final_base=base if p['beta']>0 or beta==0 else q[:,4]
        forecasts=dict(moirai_direct=q[0,4],no_rag=final_base[0])
        for kind in ['encoder_only','future_state_rag','routed']:
            supplied=deltas if kind=='routed' else {kind:deltas[kind]}
            forecasts[kind]=(final_base+apply_policy(self.policies[kind],supplied,np.array([sid]))*std)[0]
        result=dict(forecasts={k:v.tolist() for k,v in forecasts.items()},selected_starts=selected.tolist(),weights=weights.tolist(),
            family=self.policies['routed'][str(sid)]['family'],beta=self.policies['routed'][str(sid)]['beta'],length=96,horizon=96,
            foundation_frozen=True,encoder='self-trained joint future-state ridge',query_future_used=False)
        if return_components:
            ids=np.searchsorted(library,selected)
            cases=candidate_innovations(x,np.zeros(96),past[ids,:,local_sid],future[ids,:,local_sid],std)*std
            result['components']=dict(base=base[0],final_base=final_base[0],encoded_path=encoded_path,
                cases=cases,deltas={k:v[0] for k,v in deltas.items()},quantiles=q[0],std=std,
                disabled=bool(p['beta']==0 and beta>0))
        return result
