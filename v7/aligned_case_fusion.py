"""History-fitted alignment and a small forecast-conditioned five/ten-case adapter.

Inspired by Align-RAG and Cross-RAG; this is an external Moirai2 adapter,
not a reproduction of either complete paper architecture.
"""
import numpy as np
import torch
from torch import nn

MODES = ['wiener', 'legacy', 'without_retrieval']


def align_cases(history, past, future, mode='wiener', memory_std=None):
    x, p, y = [np.asarray(v, dtype=np.float64) for v in (history, past, future)]
    if x.ndim!=2 or x.shape[1]!=96 or p.ndim!=3 or p.shape[0]!=len(x) or p.shape[1] not in [5,10] or p.shape[2]!=96 or y.shape!=p.shape:
        raise ValueError('Expected query96 and five/ten paired96 cases')
    if not all(np.isfinite(v).all() for v in [x,p,y]): raise ValueError('Nonfinite alignment inputs')
    qs, ps = x.std(1), p.std(2)
    if mode=='wiener':
        denom = ps**2+(qs[:,None]/5.)**2
        slope = np.divide(qs[:,None]*ps, denom, out=np.zeros_like(ps), where=denom>1e-12)
        intercept = x.mean(1)[:,None]-slope*p.mean(2)
    elif mode=='legacy':
        floor=1e-6 if memory_std is None else .1*np.asarray(memory_std)[:,None]
        slope = np.clip(qs[:,None]/np.maximum(ps,floor),.25,4.)
        intercept = x[:,-16:].mean(1)[:,None]-slope*p[:,:,-16:].mean(2)
    else: raise ValueError('Unknown case alignment')
    return ((p*slope[:,:,None]+intercept[:,:,None]).astype(np.float32),
        (y*slope[:,:,None]+intercept[:,:,None]).astype(np.float32))


def inputs(data, stats, period, mode):
    """Select ONLY observed/history-derived fields; no raw_y argument is read."""
    x = np.asarray(data['raw_x'],dtype=np.float64)
    q = np.asarray(data['raw_quantiles'],dtype=np.float64)
    sid = np.asarray(data['sid'],dtype=int); start = np.asarray(data['start'])
    count=int(stats.get('case_count',10))
    if x.shape!=(len(sid),96) or q.shape!=(len(sid),9,96): raise ValueError('Expected raw96 and nine forecast quantiles')
    sd = np.asarray(stats['std'])[sid]; means = np.asarray(stats['mean'])[sid]
    level = x.mean(1); scale = np.maximum(x.std(1),.1*sd)
    normalized_x = (x-level[:,None])/scale[:,None]
    normalized_q = (q-level[:,None,None])/scale[:,None,None]
    lead = np.arange(96)-47.5
    phase = 2*np.pi*(start+96)/(period or 1)
    state = np.stack([(level-means)/sd, np.log(scale/sd), (x[:,-1]-level)/scale,
        (x@lead)/(lead@lead)*95/scale, np.sin(phase),np.cos(phase),np.sin(phase/7),np.cos(phase/7)],axis=1)
    if mode=='without_retrieval':
        # Keep the architecture and activate its memory pathway using five/ten identical
        # query/prediction pairs. Every value comes from this query, not a case bank.
        past = np.repeat(normalized_x[:,None],count,axis=1)
        future = np.repeat(normalized_q[:,4,None],count,axis=1)
        metadata = np.zeros((len(x),count,int(stats.get('metadata_width',5))),dtype=np.float32)
    else:
        p = np.asarray(data['case_past'],dtype=np.float64)
        y = np.asarray(data['case_future'],dtype=np.float64)
        if p.shape!=(len(x),count,96):raise ValueError('Configured case count differs from actual input')
        memory_sd=np.asarray(stats.get('memory_std',stats['std']))[sid]
        past, future = align_cases(x,p,y,mode,memory_sd)
        past = (past-level[:,None,None])/scale[:,None,None]
        future = (future-level[:,None,None])/scale[:,None,None]
        selected = np.asarray(data['selected_starts'])
        if np.any(selected+192>start[:,None]): raise ValueError('Self or overlapping retrieved episode')
        metadata = np.stack([(p.mean(2)-level[:,None])/scale[:,None],
            np.log(np.maximum(p.std(2),1e-6)/scale[:,None]),
            (p[:,:,-1]-p[:,:,0])/scale[:,None],
            np.log1p((start[:,None]-selected-192)/192.),
            np.mean((p-x[:,None])**2,2)/scale[:,None]**2],axis=2)
        if int(stats.get('metadata_width',5))==13:
            old_phase=2*np.pi*(selected+96)/(period or 1);difference=phase[:,None]-old_phase
            clocks=np.stack([np.sin(old_phase),np.cos(old_phase),np.sin(old_phase/7),np.cos(old_phase/7),
                np.sin(difference),np.cos(difference),np.sin(difference/7),np.cos(difference/7)],axis=2)
            metadata=np.concatenate([metadata,clocks],axis=2)
    radius = np.minimum(.25*np.maximum(q[:,-1]-q[:,0],0),2*scale[:,None])/scale[:,None]
    result = dict(history=normalized_x,quantiles=normalized_q,past=past,future=future,
        metadata=metadata,state=state,radius=radius,sid=sid)
    if stats.get('residual_reference'):
        result['reference']=(np.asarray(data['reference'])-level[:,None])/scale[:,None]
        result['probe_error']=(x[:,64:]-np.asarray(data['probe_quantiles'])[:,4])/scale[:,None]
        if mode=='without_retrieval':result['future']=np.repeat(result['reference'][:,None],count,axis=1)
    result = {k:(v.astype(np.int64) if k=='sid' else np.clip(v,-8,8).astype(np.float32)) for k,v in result.items()}
    return result, level.astype(np.float32), scale.astype(np.float32)


class CaseFusion(nn.Module):
    """Twelve forecast patches attend to five/ten historical cases and a null case."""
    def __init__(self, variables, width=32, conditioned=False, paired=False,metadata_width=5):
        super().__init__(); self.width=width;self.conditioned=conditioned;self.paired=paired
        self.history = nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.Linear(64,width))
        self.past = nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.Linear(64,width))
        self.state = nn.Linear(8,width); self.metadata = nn.Linear(metadata_width,width)
        self.quantiles = nn.Linear(9*8,width); self.future = nn.Linear(8,width)
        if conditioned:self.probe=nn.Linear(32,width);self.reference=nn.Linear(8,width)
        self.position = nn.Parameter(torch.randn(12,width)*.02)
        self.route = nn.Embedding(2,width); self.series = nn.Embedding(variables,width)
        self.query = nn.Linear(width,width); self.key=nn.Linear(width,width); self.value=nn.Linear(width,width)
        self.null_key=nn.Parameter(torch.zeros(1,width)); self.null_value=nn.Parameter(torch.zeros(1,width))
        self.gate=nn.Sequential(nn.Linear(width,1),nn.Sigmoid())
        self.output=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,64),nn.GELU(),nn.Linear(64,8))
        nn.init.zeros_(self.series.weight)
        nn.init.zeros_(self.output[-1].weight); nn.init.zeros_(self.output[-1].bias)

    def forward(self, history, quantiles, past, future, metadata, state, radius, sid, return_attention=False,reference=None,probe_error=None,return_scores=False):
        n=len(history); w=self.width
        h=self.history(history)+self.state(state)+self.series(sid)
        q=self.quantiles(quantiles.reshape(n,9,12,8).permute(0,2,1,3).reshape(n,12,72))+h[:,None]+self.position[None]
        if self.conditioned:q=q+self.probe(probe_error)[:,None]+self.reference(reference.reshape(n,12,8))
        count=past.shape[1];route=torch.arange(count,device=history.device)//5
        keys=self.past(past)+self.metadata(metadata)+self.route(route)[None]
        values=self.future(future.reshape(n,count,12,8))+keys[:,:,None]+self.position[None,None]
        history_keys=self.key(keys)
        if self.paired:
            projected_keys=torch.cat([history_keys,self.null_key[None].expand(n,-1,-1)],1)
            projected_values=self.value(values)
            projected_values=torch.cat([projected_values,self.null_value[None,:,None].expand(n,-1,12,-1)],1)
            logits=torch.bmm(self.query(q),projected_keys.transpose(1,2))/(w**.5)
            weights=torch.softmax(logits,dim=-1)
            attended=torch.einsum('npk,nkpd->npd',weights,projected_values)
        else:
            expanded=(keys[:,:,None]+self.position[None,None]).reshape(n,count*12,w)
            expanded_keys=torch.cat([self.key(expanded),self.null_key[None].expand(n,-1,-1)],1)
            flat_values=torch.cat([self.value(values.reshape(n,count*12,w)),self.null_value[None].expand(n,-1,-1)],1)
            logits=torch.bmm(self.query(q),expanded_keys.transpose(1,2))/(w**.5)
            weights=torch.softmax(logits,dim=-1);attended=torch.bmm(weights,flat_values)
        hidden=q+self.gate(q)*attended
        correction=torch.tanh(self.output(hidden)).reshape(n,96)*radius
        if return_scores:
            candidates=torch.cat([history_keys,self.null_key[None].expand(n,-1,-1)],1)
            query=torch.nn.functional.normalize(self.query(h),dim=-1)
            candidates=torch.nn.functional.normalize(candidates,dim=-1)
            return correction,torch.einsum('nd,nkd->nk',query,candidates)/.1
        return (correction,weights) if return_attention else correction


def tensor_inputs(values, device):
    return {k:torch.as_tensor(v,device=device) for k,v in values.items()}


@torch.no_grad()
def forecast(model, features, data, scale, batch=512):
    model.eval(); output=[]
    for i in range(0,len(scale),batch): output.append(model(**{k:v[i:i+batch] for k,v in features.items()}).cpu().numpy())
    delta=np.concatenate(output)*np.asarray(scale)[:,None]
    base=np.asarray(data['reference']) if 'reference' in data else np.asarray(data['raw_quantiles'])[:,4]
    return (base+delta).astype(np.float32)
