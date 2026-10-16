"""Audit the fast pilot; no fitting, policy changes, or TEST winner selection."""
import argparse,json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import sha256,write_json
from .benchmark_local_pair_adaptation import load,read,metric,candidates
from .local_pair_adaptation import forecasts


def cluster_interval(y,baseline,prediction,starts,period):
    b=np.mean((baseline-y)**2,1);p=np.mean((prediction-y)**2,1)
    week=(starts+96)//max(1,int(7*(period or 1)))
    losses=np.array([[b[week==k].sum(),p[week==k].sum()] for k in np.unique(week)])
    rng=np.random.default_rng(13)
    sample=losses[rng.integers(0,len(losses),size=(2000,len(losses)))].sum(1)
    return dict(gain_percent=float(100*(1-p.mean()/b.mean())),
                exploratory_interval95=np.quantile(100*(1-sample[:,1]/sample[:,0]),[.025,.975]).tolist(),
                week_groups=len(losses),note='Paired weekly clusters; reused TEST; not corrected for repeated exploration')


def verify(root):
    root=Path(root);frozen=read(root/'frozen.json');records=[]
    for name,expected in frozen.items():
        folder=root/name
        assert sha256(folder/'contract.json')==expected
        contract=read(folder/'contract.json');policies=read(folder/'policy.json')
        assert sha256(folder/'policy.json')==contract['policy_sha256']
        assert sha256(folder/'model.npz')==contract['model_sha256']
        with np.load(folder/'model.npz',allow_pickle=False) as f:model={k:f[k].copy() for k in f.files}
        with np.load(folder/'predictions.npz',allow_pickle=False) as f:saved={k:f[k].copy() for k in f.files}
        d,stats,meta=load(name,'test')
        ids=[0,len(d['sid'])-1]
        if name=='electricity':ids.append(int(np.flatnonzero(d['sid']==313)[0]))
        small={k:v[ids] for k,v in d.items()}
        original,gates=forecasts(model,small,stats,meta['period'])
        poisoned=dict(small,raw_y=np.full_like(small['raw_y'],1e15))
        changed,changed_gates=forecasts(model,poisoned,stats,meta['period'])
        for key in original:np.testing.assert_array_equal(original[key],changed[key])
        for key in gates:np.testing.assert_array_equal(gates[key],changed_gates[key])
        base=small['raw_quantiles'][:,4];gp=policies['global']['selected']
        baseline=base if gp=='reference' else base+float(gp)*(original['global_raw']-base)
        np.testing.assert_allclose(baseline,saved['no_rag'][ids],rtol=1e-12,atol=1e-9)
        for arm in ('real','checked','deranged','random','mean'):
            key=policies[arm]['selected'];forecast=baseline if key=='reference' else candidates(original,gates,baseline,arm)[key]
            np.testing.assert_allclose(forecast,saved[arm][ids],rtol=1e-12,atol=1e-9)
        store=Store(meta['store'])
        try:
            for row in ids:
                sid=int(d['sid'][row])
                for prefix in ('','random_'):
                    for j,start in enumerate(d[prefix+'selected_starts'][row]):
                        np.testing.assert_array_equal(d[prefix+'case_past'][row,j],store.window(sid,int(start),96))
                        np.testing.assert_array_equal(d[prefix+'case_future'][row,j],store.window(sid,int(start)+96,96))
        finally:store.close()
        ci=cluster_interval(d['raw_y'],saved['no_rag'],saved['checked'],d['start'],meta['period'])
        record=dict(dataset=name,replayed_queries=len(ids),physical_case_pairs_checked=20*len(ids),
                    target_poison_unchanged=True,extra_retrieval_gain=ci)
        records.append(record);print('VERIFIED LOCAL',record,flush=True)
    write_json(root/'verification.json',dict(complete=True,records=records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();verify(a.root)
