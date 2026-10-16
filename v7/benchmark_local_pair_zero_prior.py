"""Post-hoc diagnostic: adapt retrieved pairs directly from frozen Moirai.

Motivated by the first pilot's poor global-head Weather generalization. This
is a separate, uniformly specified experiment, never a dataset-wise replacement.
"""
import argparse,time,gc
from pathlib import Path
import numpy as np
from .common import sha256,write_json
from .current_forecast import DATASETS
from .local_pair_adaptation import train_global,forecasts
from .benchmark_local_pair_adaptation import load,read,select,candidates,evaluate


def fit_zero_prior(data,stats,period):
    # No TRAIN future labels influence this model: zero residual targets, PCA
    # from historical inputs only. The closed-form local fit uses OLD cases.
    model=train_global(dict(data,raw_y=data['raw_quantiles'][:,4]),stats,period)
    np.testing.assert_array_equal(model['prior'],0.)
    return model


def calibrate(name,out):
    train,stats,meta=load(name,'train');model=fit_zero_prior(train,stats,meta['period'])
    del train;gc.collect();d,_,_=load(name,'validation')
    origins=np.unique(d['start']);chunks=np.array_split(origins,3)
    cal=np.flatnonzero((d['start']<chunks[2][0])&(d['start']+192<=chunks[2][0]))
    catalog=read(Path(meta['store'])/'catalog.json');test_start=min(s['validation_end'] for s in catalog['series'])
    confirm=np.flatnonzero(np.isin(d['start'],chunks[2])&(d['start']+192<=test_start))
    if not len(cal) or not len(confirm):raise ValueError('Empty purged calibration')
    outputs,gates=forecasts(model,d,stats,meta['period']);base=d['raw_quantiles'][:,4]
    policies={'global':dict(selected='reference',proposed=None,confirmed=False)}
    for arm in ('real','checked','deranged','random','mean'):
        policies[arm]=select(d,candidates(outputs,gates,base,arm),base,stats,cal,confirm)
    out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'model.npz',**model)
    write_json(out/'policy.json',policies)
    write_json(out/'contract.json',dict(dataset=name,primary='checked',zero_prior=True,
        model_sha256=sha256(out/'model.npz'),policy_sha256=sha256(out/'policy.json'),
        code_sha256={f:sha256('v7/'+f) for f in ['local_pair_adaptation.py','benchmark_local_pair_adaptation.py','benchmark_local_pair_zero_prior.py']},
        prior='Exactly zero; baseline is direct frozen Moirai. PCA8 uses TRAIN histories only.',
        rule='Same penalties, strengths, purged VAL selection and confirmation as first pilot; all datasets frozen before this TEST phase',
        cal_origins=np.unique(d['start'][cal]).tolist(),confirm_origins=np.unique(d['start'][confirm]).tolist(),
        caveats=['Post-hoc supplement motivated by first pilot; not a predeclared primary',
                 'Old retrieval choices reused VAL; previously explored TEST',
                 'Not full meta-training or current-history prefix validation']))
    print('ZERO FROZEN',name,{k:v['selected'] for k,v in policies.items()},flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS));a=p.parse_args()
    root=Path(a.output)
    if root.exists():raise FileExistsError('Choose a fresh supplemental directory')
    root.mkdir(parents=True);begin=time.perf_counter()
    for name in a.datasets:calibrate(name,root/name)
    write_json(root/'frozen.json',{name:sha256(root/name/'contract.json') for name in a.datasets})
    records=[evaluate(name,root/name) for name in a.datasets]
    write_json(root/'summary.json',dict(complete=True,primary='checked',supplemental=True,
        seconds=time.perf_counter()-begin,records=records,
        caveat='Post-hoc zero-prior diagnostic; original pilot and current default are preserved'))


if __name__=='__main__':main()
