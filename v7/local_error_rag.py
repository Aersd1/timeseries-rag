"""Causal error-case memory and local calibration without a global utility head."""
import numpy as np
from .trend_rag import decompose,normalize
from .state_recall import history_state,state_distance
from .rerank import future_scores,aggregate
from .learned_rerank import diverse_selection
from .benefit_rerank import select_candidates,guarded_policy

PARAMETERS=[dict(mode='compatibility',weight=w,anchors=8) for w in [0.,.2,1.]]+[
    dict(mode='local_benefit',weight=w,anchors=k) for w in [0.,.2,1.] for k in [4,8]]


def independent_nearest(scores,starts,limit):
    selected=[]
    for i in np.lexsort((starts,scores)):
        if all(abs(int(starts[i])-int(starts[j]))>=192 for j in selected):selected.append(int(i))
        if len(selected)==limit:break
    return np.asarray(selected,dtype=int)


class LocalErrorRAG:
    def __init__(self,pool,raw_past,errors,memory_std,period,parameters=PARAMETERS):
        self.pool=pool;self.errors=np.asarray(errors,dtype=np.float32)
        self.std=max(float(memory_std),1e-6);self.period=period;self.parameters=parameters
        self.states=history_state(raw_past,self.std);self.observed=[]
        if self.errors.shape!=(len(pool.starts),96):raise ValueError('Memory error alignment mismatch')

    def observe(self,start,history,future,base,potentials,clock):
        if start+192>clock:raise ValueError('Episode is not fully observed before current history')
        if any(x['start']==start for x in self.observed):raise ValueError('Duplicate observation')
        x=np.asarray(history,dtype=np.float32);y=np.asarray(future,dtype=np.float32)
        if x.shape!=(96,) or y.shape!=(96,) or np.asarray(potentials).shape!=(len(self.parameters),96):
            raise ValueError('Wrong completed episode shape')
        _,r=decompose(np.r_[x,y],25);past,mean,scale=normalize(r[:96],self.pool.floor)
        self.observed.append(dict(start=int(start),past=past,future=(r[96:]-mean)/scale,
            state=history_state(x,self.std),error=(y-np.asarray(base,dtype=np.float32))/self.std,
            potentials=np.asarray(potentials,dtype=np.float32)))

    def predict(self,history,quantiles,start):
        """No current true future accepted; all feedback must already be complete."""
        x=np.asarray(history,dtype=np.float32);quantiles=np.asarray(quantiles,dtype=np.float32)
        if x.shape!=(96,) or quantiles.shape!=(9,96) or not np.isfinite(quantiles).all():raise ValueError('Invalid history/forecast')
        if any(r['start']+192>start for r in self.observed):raise ValueError('Observation travels backwards in time')
        p=self.pool;eligible=np.flatnonzero(p.starts+192<=min(start,p.memory_end))
        known=self.observed
        starts=np.r_[p.starts[eligible],[r['start'] for r in known]].astype(np.int64)
        if len(starts)<100:raise ValueError('Need100 complete cases')
        if len(np.unique(starts))!=len(starts):raise ValueError('Duplicate memory positions')
        past=np.concatenate([p.past[eligible],np.array([r['past'] for r in known],dtype=np.float32).reshape(-1,96)])
        states=np.concatenate([self.states[eligible],np.array([r['state'] for r in known]).reshape(-1,3)])
        values=np.concatenate([self.errors[eligible],np.array([r['error'] for r in known],dtype=np.float32).reshape(-1,96)])
        future=np.concatenate([p.future[eligible],np.array([r['future'] for r in known],dtype=np.float32).reshape(-1,96)])
        _,residual=decompose(x,25);q,mean,scale=normalize(residual,p.floor)
        shape=np.mean((past-q)**2,axis=1)
        state=state_distance(history_state(x,self.std),states,start,starts,self.period)
        paths=np.c_[np.repeat(x[None],9,axis=0),quantiles]
        _,predicted=decompose(paths,25);projected=(predicted[:,96:]-mean)/scale
        weights=sorted(set(h['weight'] for h in self.parameters));shortlists={}
        for w in weights:shortlists[w]=np.lexsort((starts,shape if w==0 else shape+w*state))[:100]
        union=np.unique(np.concatenate(list(shortlists.values())))
        fs=np.full(len(starts),np.nan);fs[union]=future_scores(projected,np.arange(1,10)/10,future[union],0.,1.,[96])
        potentials=[];betas=[];selections=[];recent_counts=[]
        for j,h in enumerate(self.parameters):
            ids=shortlists[h['weight']];hs=shape[ids];f=fs[ids]
            prior=.8*f/max(float(np.median(f)),1e-6)+.2*hs/max(float(np.median(hs)),1e-6)
            if known:
                idx=np.arange(len(eligible),len(starts));age=(start-starts[idx]-192)/max(p.memory_end/4,192)
                scores=shape[idx]+h['weight']*state[idx]+.1*age
                anchors=independent_nearest(scores,starts[idx],h['anchors'])
                calibration=independent_nearest(scores,starts[idx],12)
            else:idx=anchors=calibration=np.empty(0,dtype=int);scores=np.empty(0)
            if h['mode']=='local_benefit':
                if len(anchors)>=3:
                    estimate,_=aggregate(values[idx[anchors]],scores[anchors])
                    covariance=np.mean(values[ids].astype(float)*estimate,axis=1)
                else:covariance=np.zeros(100)
                chosen=select_candidates(covariance,values[ids],prior,starts[ids],p.sid)
                correction=chosen['correction'];selected=chosen['selected']
            else:
                hits=[dict(sid=p.sid,start=int(a),distance=float(b)) for a,b in zip(starts[ids],hs)]
                selected=diverse_selection(hits,prior,5,192,1.)
                correction,_=aggregate(values[ids[selected]],prior[selected])
            beta=0.
            if len(calibration)>=9:
                references=[known[k] for k in calibration]
                policy=guarded_policy(np.stack([r['error'] for r in references]),
                    np.stack([r['potentials'][j] for r in references]),np.zeros(len(references),dtype=int),
                    np.array([r['start'] for r in references]))
                beta=policy['0']['beta']
            potentials.append(correction);betas.append(beta);selections.append(starts[ids[selected]])
            recent_counts.append(int(np.sum(ids>=len(eligible))))
        return dict(potentials=np.asarray(potentials,dtype=np.float32),beta=np.asarray(betas,dtype=np.float32),
            selected_starts=np.asarray(selections),recent_candidates=np.asarray(recent_counts),
            eligible_cases=len(starts),observed_queries=len(known))
