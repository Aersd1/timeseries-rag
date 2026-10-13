"""Use an observed within-history forecast error to condition RAG selection."""
import numpy as np
from .contextual_adapter import feature as contextual_feature
from .learned_rerank import diverse_selection
from .rerank import aggregate
from .benefit_rerank import fit_ridge

WEIGHTS=[0.,.25,1.]


def fit_completed(features,targets,starts,clock,penalty):
    """Only complete, nonoverlapping episodes are eligible for causal updates."""
    eligible=np.asarray(starts)+192<=clock
    if eligible.sum()<4:raise ValueError('Insufficient completed update episodes')
    return fit_ridge(features[eligible].reshape(-1,features.shape[-1]),targets[eligible].reshape(-1),penalty)


def rerank(data,ids,known_error,memory_starts,memory_errors,weight):
    selected=[];weights=[];errors=[]
    for row,i in enumerate(ids):
        starts=data['candidate_starts'][i]
        if np.any(starts+192>data['start'][i]):raise ValueError('Overlapping candidate')
        positions=np.searchsorted(memory_starts,starts)
        np.testing.assert_array_equal(memory_starts[positions],starts)
        e=memory_errors[positions]
        observed=known_error[row]/data['std'][i]
        mismatch=(e[:,:32].mean(1)-observed.mean())**2
        # A floor avoids overconfident matching when the32-point probe is flat.
        mismatch/=max(float(observed.std()**2),.1)
        fs=data['future_scores'][i];hs=data['history_scores'][i]
        score=.8*fs/max(float(np.median(fs)),1e-6)+.2*hs/max(float(np.median(hs)),1e-6)+weight*mismatch
        hits=[dict(sid=int(data['sid'][i]),start=int(p),distance=float(h)) for p,h in zip(starts,hs)]
        chosen=diverse_selection(hits,score,5,192,1.)
        _,w=aggregate(e[chosen],score[chosen])
        selected.append(starts[chosen]);weights.append(w);errors.append(e[chosen])
    return dict(selected_starts=np.asarray(selected),weights=np.asarray(weights),errors=np.asarray(errors))


def feature(store,data,ids,s,period,plan,known_error,use_rag=True,reranked=None):
    # Slice first so patched candidate records cannot alter another query.
    current={k:data[k][ids] for k in ['raw_x','base','raw_quantiles','start','std','weights','errors','selected_starts']}
    if reranked is not None:current.update(reranked)
    f=contextual_feature(store,current,np.arange(len(ids)),s,period,plan,use_rag)
    e=np.asarray(known_error,dtype=float)/current['std'][:,None]
    if e.shape!=(len(ids),32):raise ValueError('Expected32 observed probe errors')
    lead=np.arange(1,97)[None,:]/96;t=np.arange(32)-15.5
    values=[e.mean(1),e[:,-8:].mean(1),e.std(1),e[:,:8].mean(1),e[:,8:16].mean(1),e[:,16:24].mean(1)]
    extra=[np.broadcast_to(v[:,None],(len(ids),96)) for v in values]
    extra.extend([e.mean(1)[:,None]*lead,(e@t)[:,None]/(t@t)*32*lead])
    if 'scale_ratio' in data:
        ratio=np.asarray(data['scale_ratio'][ids],dtype=float)
        extra.extend([np.broadcast_to(ratio[:,None],(len(ids),96)),np.broadcast_to(np.log(np.maximum(ratio,1e-6))[:,None],(len(ids),96))])
    if data.get('annual_calendar',False):
        annual=2*np.pi*(current['start'][:,None]+96+np.arange(96)[None,:])/(365.2425*(period or 1))
        daily=2*np.pi*(current['start'][:,None]+96+np.arange(96)[None,:])/(period or 1)
        extra.extend([np.sin(annual),np.cos(annual),np.sin(annual)*np.sin(daily),np.sin(annual)*np.cos(daily),
            np.cos(annual)*np.sin(daily),np.cos(annual)*np.cos(daily)])
        if use_rag:
            mean_error=(current['weights'][:,:,None]*current['errors']).sum(1).mean(1,keepdims=True)
            extra.extend([mean_error*np.sin(annual),mean_error*np.cos(annual)])
    return np.concatenate([f,np.stack(extra,axis=-1).astype(np.float32)],axis=-1)
