"""Synchronous weekly-cluster uncertainty and prespecified temporal stability."""
import numpy as np


def compare(base,forecast,truth,starts,period,replicates=5000):
    a=np.mean((np.asarray(base,dtype=float)-np.asarray(truth,dtype=float))**2,1)
    b=np.mean((np.asarray(forecast,dtype=float)-np.asarray(truth,dtype=float))**2,1)
    starts=np.asarray(starts);groups=np.floor((starts-starts.min())/(7*float(period))).astype(int)
    blocks=np.array([[a[groups==g].sum(),b[groups==g].sum(),np.sum(groups==g)] for g in np.unique(groups)])
    rng=np.random.default_rng(20261015);boot=blocks[rng.integers(len(blocks),size=(replicates,len(blocks)))].sum(1)
    bg=100*(boot[:,0]-boot[:,1])/np.maximum(boot[:,0],1e-30)
    ci=np.quantile(bg,[.025,.975]).tolist();family=np.quantile(bg,[.05/(2*7),1-.05/(2*7)]).tolist()
    windows=[]
    for part in np.array_split(np.unique(starts),4):
        ids=np.isin(starts,part);aa=float(a[ids].mean());bb=float(b[ids].mean())
        windows.append(dict(first_start=int(part[0]),last_start=int(part[-1]),queries=int(ids.sum()),base_mse=aa,mse=bb,gain_percent=float(100*(aa-bb)/max(aa,1e-30))))
    gain=float(100*(a.mean()-b.mean())/max(a.mean(),1e-30))
    return dict(gain_percent=gain,weekly_clusters=len(blocks),interval95=ci,seven_dataset_familywise_interval=family,
        time_blocks=windows,meets_prespecified_stability=bool(ci[0]>0 and all(w['gain_percent']>0 for w in windows)),
        exploratory=True)


def adoption_diagnostic(base,candidate,forecast,mask,truth,sid,strength,replicates=200):
    base=np.asarray(base).reshape(-1,12,8);candidate=np.asarray(candidate).reshape(-1,12,8);truth=np.asarray(truth,dtype=float).reshape(-1,12,8)
    before=np.mean((base.astype(float)-truth)**2,2);after=np.mean((np.asarray(forecast,dtype=float).reshape(-1,12,8)-truth)**2,2);gain=before-after
    active=int(mask.sum());rng=np.random.default_rng(20261015);random=[];per_sid=[np.flatnonzero(sid==s) for s in np.unique(sid)]
    # Same adopted counts for every variable and forecast patch; randomize origins.
    for _ in range(replicates):
        shuffled=mask.copy()
        for ids in per_sid:
            order=np.argsort(rng.random((len(ids),12)),axis=0,kind='stable')
            shuffled[ids]=np.take_along_axis(mask[ids],order,axis=0)
        pred=base.astype(float)+shuffled[:,:,None]*strength*(candidate.astype(float)-base)
        random.append(float(np.mean((pred-truth)**2)))
    return dict(adopted_fraction=float(mask.mean()),adopted_help_fraction=float((gain[mask]>0).mean()) if active else None,
        gain_contribution=float(np.maximum(gain,0).mean()),harm_contribution=float(np.maximum(-gain,0).mean()),
        matched_coverage_random_mse=float(np.mean(random)),randomization_interval=np.quantile(random,[.025,.975]).tolist(),
        note='Adoption means the real-case forecast branch was applied, not proof that each supplied case helped')
