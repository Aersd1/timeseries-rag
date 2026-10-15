"""Case-history + continuation increments around an independently frozen base.

No current target is read by pack/summary/predict. An empty reference removes
EVERY case input and returns the supplied base bit-for-bit. History is used in
both attention keys and regression values, not merely for retrieval matching.
"""
import numpy as np
import torch
from torch import nn
from .unified_residual import features

ARMS = ['paired', 'history_only', 'without_history_sequence', 'deranged', 'matched_random']
PLAN = dict(seed=13, epochs=16, batch_size=256, width=32, learning_rate=.001,
    weight_decay=.001, correction_energy=.001, null_energy=.01,
    whole_reference_dropout=.25, history_dropout=.10, continuation_dropout=.10,
    per_case_dropout=.10, baseline_modes=['moirai', 'frozen_query_adapter'],
    utility_family='summary_mlp', utility_seeds=[13,41,73], utility_epochs=20,
    query_length=96, horizon=96, cases=10,
    primary='paired', selection='VAL first two purged blocks choose; last two only confirm',
    limitations=['Without-history ablation removes explicit96 sequence and metadata; continuation alignment still uses historical mean/std.',
        'Previously explored TEST; one upstream increment seed, not an independent blind benchmark.',
        'Frozen query adapter is prefix-fitted; its residuals on increment-fit rows are in-sample. Utility labels are held-forward.'])


def pack(data, stats, period, arm='paired'):
    if arm not in ARMS+['empty']:raise ValueError('Unknown reference arm')
    mode = 'query_only' if arm=='empty' else 'history_only' if arm=='history_only' else 'matched_random' if arm=='matched_random' else 'real'
    f, _, scale = features(data, stats, period, mode)
    n=len(scale); p={k:f[k] for k in ['history','quantiles','state','probe_error','radius']}
    p.update(past=np.zeros((n,10,96),np.float32), continuation=np.zeros((n,10,12,24),np.float32),
        metadata=np.zeros((n,10,13),np.float32), past_present=np.zeros((n,10,1),np.float32),
        future_present=np.zeros((n,10,1),np.float32), valid=np.zeros((n,10),np.float32))
    if arm=='empty':return p,scale
    prefix='random_' if arm=='matched_random' else ''
    selected=data[prefix+'selected_starts']
    if np.any(selected+192>data['start'][:,None]):raise ValueError('Self/overlapping reference')
    p['valid'][:]=1
    if arm!='without_history_sequence':
        p['past']=f['past'];p['metadata']=f['metadata'][:,:,:13];p['past_present'][:]=1
    if arm!='history_only':
        # The entire observed continuation AND its frozen forecast AND its error.
        # The scalar alignment is kept identical in the explicit-history ablation.
        past=data[prefix+'case_past'];future=data[prefix+'case_future'];prior=data[prefix+'case_baseline']
        for begin in range(0,n,256):
            sl=slice(begin,min(begin+256,n));x=data['raw_x'][sl].astype(float);old=past[sl].astype(float)
            qs=x.std(1);ps=old.std(2);den=ps**2+(qs[:,None]/5.)**2
            slope=np.divide(qs[:,None]*ps,den,out=np.zeros_like(ps),where=den>1e-12)
            mean=old.mean(2);ss=scale[sl,None,None]
            raw=(future[sl].astype(float)-mean[:,:,None])*slope[:,:,None]/ss
            pred=(prior[sl].astype(float)-mean[:,:,None])*slope[:,:,None]/ss
            residual=(future[sl].astype(float)-prior[sl].astype(float))*slope[:,:,None]/ss
            p['continuation'][sl]=np.clip(np.concatenate([v.reshape(-1,10,12,8) for v in [raw,pred,residual]],3),-8,8)
        p['future_present'][:]=1
    if arm=='deranged':
        p['continuation']=p['continuation'][np.arange(n)[:,None],data['shuffle_order']]
    if not all(np.isfinite(v).all() for v in p.values()):raise ValueError('Nonfinite observed input')
    return p,scale


def with_base(packed, base, data, scale, kind=0):
    p=dict(packed);level=np.asarray(data['raw_x'],float).mean(1)
    p['base']=np.clip((np.asarray(base,float)-level[:,None])/scale[:,None],-8,8).astype(np.float32)
    p['base_kind']=np.full((len(base),1),kind,np.float32)
    return p


class PairedHistoryIncrement(nn.Module):
    def __init__(self,width=32):
        super().__init__();w=width
        self.history=nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.Linear(64,w))
        self.past=nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.Linear(64,w))
        self.state=nn.Linear(8,w);self.probe=nn.Linear(32,w);self.meta=nn.Linear(13,w)
        self.baseline_kind=nn.Linear(1,w,bias=False)
        self.forecast=nn.Linear(80,w)
        self.future=nn.Sequential(nn.Linear(24,64),nn.GELU(),nn.Linear(64,w))
        self.pair_value=nn.Sequential(nn.Linear(2*w,w),nn.GELU(),nn.Linear(w,w))
        self.route=nn.Embedding(2,w);self.position=nn.Parameter(torch.randn(12,w)*.02)
        self.key=nn.Linear(w,w);self.query=nn.Linear(w,w)
        self.output=nn.Sequential(nn.LayerNorm(2*w),nn.Linear(2*w,64),nn.GELU(),nn.Linear(64,8))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)

    def forward(self,history,quantiles,state,probe_error,radius,past,continuation,metadata,
                past_present,future_present,valid,base,base_kind,return_null=False):
        n=len(history);w=self.position.shape[1]
        quant=quantiles.reshape(n,9,12,8).permute(0,2,1,3).reshape(n,12,72)
        q=self.forecast(torch.cat([quant,base.reshape(n,12,8)],2))+self.position[None]
        q=q+(self.history(history)+self.state(state)+self.probe(probe_error)+self.baseline_kind(base_kind))[:,None]
        # past96 enters every case VALUE as well as the attention KEY.
        ph=(self.past(past)+self.meta(metadata))*past_present
        fut=self.future(continuation)*future_present[:,:,None]
        route=self.route(torch.arange(10,device=q.device)//5)[None]
        keys=self.key(ph+fut.mean(2)+route)
        pair=torch.cat([ph[:,:,None].expand(-1,-1,12,-1),fut],3)
        values=self.pair_value(pair)*valid[:,:,None,None]
        logits=torch.einsum('npd,nkd->npk',self.query(q),keys)/(w**.5)
        logits=logits.masked_fill(valid[:,None]==0,-1e4)
        logits=torch.cat([logits,torch.zeros(n,12,1,device=q.device,dtype=q.dtype)],2)
        weights=torch.softmax(logits,2)[:,:,:10]
        context=torch.einsum('npk,nkpd->npd',weights,values)
        empty=torch.zeros_like(context)
        full=self.output(torch.cat([q,context],2))
        null=self.output(torch.cat([q,empty],2))
        delta=((torch.tanh(full)-torch.tanh(null)).reshape(n,96)*radius)
        delta=torch.maximum(torch.minimum(delta,radius),-radius)
        return (delta,null) if return_null else delta


def mask_modalities(p, history=False, future=False, empty=False):
    out=dict(p)
    if history or empty:
        for k in ['past','metadata','past_present']:out[k]=np.zeros_like(p[k])
    if future or empty:
        for k in ['continuation','future_present']:out[k]=np.zeros_like(p[k])
    if empty:out['valid']=np.zeros_like(p['valid'])
    return out


def fit_increment(data, bases, stats, period, arm, device='cuda'):
    torch.manual_seed(PLAN['seed']);rng=np.random.default_rng(PLAN['seed'])
    p,scale=pack(data,stats,period,arm);n=len(scale)
    model=PairedHistoryIncrement().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    sd=np.asarray(stats['std'])[data['sid']];weight=.5*(scale/sd)**2+.5*scale**2/np.mean(np.asarray(stats['std'])**2)
    level=data['raw_x'].astype(float).mean(1);losses=[]
    for _ in range(PLAN['epochs']):
        model.train();order=rng.permutation(n);total=0.
        for begin in range(0,n,PLAN['batch_size']):
            ids=order[begin:begin+PLAN['batch_size']];kind=rng.integers(2,size=len(ids))
            b=np.where(kind[:,None]==0,bases[0][ids],bases[1][ids])
            batch={k:torch.as_tensor(v[ids],device=device) for k,v in p.items()}
            batch['base']=torch.as_tensor(np.clip((b-level[ids,None])/scale[ids,None],-8,8).astype(np.float32),device=device)
            batch['base_kind']=torch.as_tensor(kind[:,None],dtype=torch.float32,device=device)
            null=rng.random(len(ids))<.25;hp=rng.random(len(ids))<.10;fp=rng.random(len(ids))<.10
            keep=~(null[:,None]|(rng.random((len(ids),10))<.10))
            batch['valid']=batch['valid']*torch.as_tensor(keep,dtype=torch.float32,device=device)
            batch['past_present']=batch['past_present']*torch.as_tensor(~hp[:,None,None],device=device)
            batch['future_present']=batch['future_present']*torch.as_tensor(~fp[:,None,None],device=device)
            target=torch.as_tensor(((data['raw_y'][ids].astype(float)-b)/scale[ids,None]).astype(np.float32),device=device)
            delta,empty=model(**batch,return_null=True);e=delta-target
            ww=torch.as_tensor(weight[ids],dtype=torch.float32,device=device)
            loss=((e.square()+.02*e.abs()).mean(1)*ww).mean()+.001*delta.square().mean()+.01*empty.square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite paired increment')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(loss.detach())*len(ids)
        losses.append(total/n)
    return model.eval(),losses


@torch.no_grad()
def predict(model,p,base,scale,device='cuda',batch=256):
    result=[];model.eval()
    for begin in range(0,len(base),batch):
        f={k:torch.as_tensor(v[begin:begin+batch],device=device) for k,v in p.items()}
        result.append(model(**f).cpu().numpy())
    delta=np.concatenate(result)*scale[:,None]
    return (base+delta).astype(np.float32)


def summary(p,delta):
    """Observed features only; no historical/current target access."""
    n=len(delta);tile=lambda v:np.broadcast_to(v[:,None],(n,12,v.shape[-1]))
    hist=p['past']*p['past_present'];fut=p['continuation']*p['future_present'][:,:,None]
    md=p['metadata']*p['past_present'];probe=p['probe_error']
    pr=np.stack([probe.mean(1),probe.std(1),abs(probe).mean(1),probe[:,-8:].mean(1)],1)
    pieces=[tile(p['state']),p['history'].reshape(n,12,8),p['base'].reshape(n,12,8),
        p['quantiles'][:,[0,4,8]].reshape(n,3,12,8).transpose(0,2,1,3).reshape(n,12,24),
        delta.reshape(n,12,8),tile(pr),hist.mean(1).reshape(n,12,8),hist.std(1).reshape(n,12,8),
        fut[:,:,:,:8].mean(1),fut[:,:,:,16:].mean(1),fut[:,:,:,16:].std(1),tile(md.mean(1)),tile(md.std(1)),
        tile(np.concatenate([p['past_present'].mean(1),p['future_present'].mean(1),p['base_kind']],1))]
    return np.clip(np.concatenate(pieces,2),-16,16).astype(np.float32)
