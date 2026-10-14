"""Recall from observed future prototypes, using a history-only predicted state."""
import numpy as np
from .trend_rag import decompose,normalize,ResidualPool
from .rerank import future_scores,aggregate
from .learned_rerank import diverse_selection
from .future_state_encoder import history_features,prototype_state,predict
from .query_future_state_rag import FutureStatePredictor
from .error_correction import retrieve


def recall(pool,query,quantiles,query_code,prototypes,precision,start,hybrid=False):
    ids=np.flatnonzero(pool.starts+192<=min(start,pool.memory_end))
    if len(ids)<100:raise ValueError('Insufficient complete nonoverlapping cases')
    hs=np.mean((pool.past[ids]-query)**2,axis=1)
    w=prototypes['mask'][ids]*precision
    distance=np.sum((prototypes['values'][ids]-query_code)**2*w,axis=1)/np.maximum(w.sum(1),1e-6)
    score=.8*distance/max(float(np.median(distance)),1e-6)+.2*hs/max(float(np.median(hs)),1e-6)
    future_order=np.lexsort((pool.starts[ids],hs,score))
    if hybrid:
        historical=np.lexsort((pool.starts[ids],hs))[:50];chosen=list(historical);seen=set(chosen)
        for j in future_order:
            if int(j) not in seen:chosen.append(int(j));seen.add(int(j))
            if len(chosen)==100:break
        ordered=np.asarray(chosen,dtype=int)
    else:ordered=future_order[:100]
    ids=ids[ordered];hs=hs[ordered]
    fs=future_scores(quantiles,[.1,.2,.3,.4,.5,.6,.7,.8,.9],pool.future[ids],0.,1.,[96])
    compatibility=.8*fs/max(float(np.median(fs)),1e-6)+.2*hs/max(float(np.median(hs)),1e-6)
    hits=[dict(sid=pool.sid,start=int(p),distance=float(h)) for p,h in zip(pool.starts[ids],hs)]
    chosen=diverse_selection(hits,compatibility,5,192,1.);_,weights=aggregate(pool.future[ids[chosen]],compatibility[chosen])
    return dict(candidate_starts=pool.starts[ids],history_scores=hs,future_scores=fs,
        selected_starts=pool.starts[ids[chosen]],weights=weights.astype(np.float32))


class ConditionalPredictor(FutureStatePredictor):
    def __init__(self,store_path,folder,scope='prefix',device='cuda',adapter=None,recall_mode='conditional'):
        if scope not in ['prefix','train']:raise ValueError('Expected prefix or train frozen memory')
        if recall_mode not in ['conditional','history','hybrid']:raise ValueError('Unknown recall mode')
        self.scope=scope;self.recall_mode=recall_mode;self.banks={};super().__init__(store_path,folder,device,adapter)

    def memory_limit(self,series):return series['memory_end'] if self.scope=='prefix' else series['train_end']

    def candidate_metadata(self,x,sid,start,quantiles,encoder,group):
        if sid not in self.banks:
            if len(self.banks)>=4:self.banks.pop(next(iter(self.banks)))
            pool=ResidualPool(self.store,sid,self.meta['window'],end=self.memory_limit(self.store.series[sid]),stride=self.meta.get('memory_stride',1))
            if self.recall_mode!='history':
                ep=np.stack([self.joint_window(int(p),192,group) for p in pool.starts])
                prototypes=prototype_state(ep[:,:96],ep[:,96:],self.means[group],self.stds[group],int(encoder.get('block_width',16)))
            else:prototypes=None
            self.banks[sid]=(pool,prototypes)
        pool,prototypes=self.banks[sid]
        if self.recall_mode=='history':return retrieve(x,quantiles,pool,start,self.meta['window'])[1]
        joint=self.joint_window(start,96,group);joint[:,group.index(sid)]=x
        code=predict(encoder,history_features(joint[None],np.array([start]),self.means[group],self.stds[group],
            self.base_run['period'] or 1,weekly=self.grouped,block_width=int(encoder.get('history_width',16))))[0]
        _,residual=decompose(x,self.meta['window']);q,mean,scale=normalize(residual,pool.floor)
        paths=np.concatenate([np.repeat(x[None],9,axis=0),quantiles],axis=1)
        _,forecast_residual=decompose(paths,self.meta['window']);rq=(forecast_residual[:,96:]-mean)/scale
        return recall(pool,q,rq,code,prototypes,encoder['precision'],start,hybrid=self.recall_mode=='hybrid')
