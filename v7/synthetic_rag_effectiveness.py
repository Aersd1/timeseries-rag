"""Paired mechanism checks, NOT a synthetic Moirai performance claim."""
import numpy as np
from .rag_effectiveness import transfer


def run(seeds=20,queries=256,memory=512):
    cells={};identical_keys=True
    for seed in range(seeds):
        rng=np.random.default_rng(20261019+seed)
        theta=rng.uniform(-1,1,memory+queries)
        h=np.arange(96)/95;future_t=1+np.arange(96)/95
        x=.4*np.sin(2*np.pi*h)[None]+theta[:,None]*(h[None]-.5)
        x+=rng.normal(0,.03,x.shape)
        deterministic=.4*np.sin(2*np.pi*future_t)[None]+theta[:,None]*(future_t[None]-.5)
        motif=np.sin(np.pi*(np.arange(96)+.5)/96)
        unobserved=rng.choice([-1.,1.],len(theta));u=rng.random(len(theta))
        noise=rng.normal(0,.1,(len(theta),96))
        qx=x[memory:];library=x[:memory]
        raw=((qx[:,None]-library[None])**2).mean(2)
        qz=(qx-qx.mean(1,keepdims=True))/qx.std(1,keepdims=True)
        pz=(library-library.mean(1,keepdims=True))/library.std(1,keepdims=True)
        shape=((qz[:,None]-pz[None])**2).mean(2)
        chosen=[]
        for row in range(queries):
            first=np.argsort(raw[row],kind='stable')[:5]
            rest=[j for j in np.argsort(shape[row],kind='stable') if j not in set(first)][:5]
            chosen.append(np.r_[first,rest])
        chosen=np.array(chosen);past=library[chosen];starts=chosen*192
        query_start=np.full(queries,memory*192+4096);sd=np.ones(queries)
        reference_keys=chosen.copy()
        for repeatability in (0.,.5,1.):
            state=np.where(u<repeatability,np.where(theta>=0,1.,-1.),unobserved)
            y=deterministic[memory:]+state[memory:,None]*motif+noise[memory:]
            for stale in (False,True):
                library_state=state[:memory]*(-1 if stale else 1)
                case_future=deterministic[:memory][chosen]+library_state[chosen,None]*motif+noise[:memory][chosen]
                prior=deterministic[:memory][chosen]
                for skill in (0.,.9,1.):
                    base=deterministic[memory:]+skill*state[memory:,None]*motif
                    quantiles=np.repeat(base[:,None],9,axis=1)
                    probe=np.repeat(qx[:,None,64:],9,axis=1)
                    out=transfer(qx,quantiles,probe,np.zeros(queries,dtype=int),query_start,
                        past,case_future,prior,starts,sd,base,96,with_features=False)
                    b=((base-y)**2).mean();r=((out['residual']-y)**2).mean()
                    key=f'repeatability={repeatability},stale={stale},base_skill={skill}'
                    cells.setdefault(key,[]).append([b,r])
                    identical_keys &= np.array_equal(chosen,reference_keys)
    result=[]
    for key,loss in cells.items():
        values=np.array(loss);rng=np.random.default_rng(41)
        sampled=values[rng.integers(len(values),size=(4000,len(values)))].sum(1)
        gain=100*(1-sampled[:,1]/sampled[:,0])
        result.append(dict(condition=key,gain_mse_percent=float(100*(1-values[:,1].sum()/values[:,0].sum())),
            paired_seed_interval95=np.quantile(gain,[.025,.975]).tolist(),seeds=len(values)))
    return dict(seeds=seeds,queries_per_seed=queries,memory_cases=memory,paired_identical_histories_and_retrieval=bool(identical_keys),
        results=result,caveat='Known toy deterministic forecast plus latent residual; NOT Moirai. Pair/history distributions, noise draws, nearest indices and fusion are fixed across interventions; no evidence that all real-data mechanisms match this generator')
