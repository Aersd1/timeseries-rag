"""Rebuild physical pairing, all forecasts/labels, held-dataset fits and reports."""
import argparse,gc
from pathlib import Path
import numpy as np
from .common import sha256,write_json
from .benchmark_memory_access import read
from .benchmark_rag_effectiveness import load,compute,summarize
from .rag_effectiveness import FEATURES,QUERY_FEATURES,fit_sign_model
from .verify_history_length import equal
from .synthetic_rag_effectiveness import run as synthetic


def close(a,b):
    if isinstance(b,dict):
        if a.keys()!=b.keys():raise AssertionError('Changed result fields')
        for k,v in b.items():close(a[k],v)
    elif isinstance(b,list):
        if len(a)!=len(b):raise AssertionError('Changed list length')
        for x,y in zip(a,b):close(x,y)
    elif isinstance(b,(int,float)) and not isinstance(b,bool):np.testing.assert_allclose(a,b,rtol=1e-7,atol=1e-8)
    elif a!=b:raise AssertionError((a,b))


def verify(root):
    summary=read(root/'summary.json');frozen=read(root/'frozen.json');names=summary['protocol']['datasets']
    if not summary['complete']:raise ValueError('Incomplete experiment')
    validation={};physical=0
    for name in names:
        folder=root/name
        for file,h in frozen[name].items():
            if sha256(folder/file)!=h:raise AssertionError('Changed frozen threshold or model')
        d,evidence=load(name,'validation');features,preds,losses,error=compute(d,evidence);physical+=evidence['physical_case_pairs']
        with np.load(folder/'validation_features.npz',allow_pickle=False) as f:
            np.testing.assert_allclose(f['features'],features,rtol=1e-12,atol=1e-12)
            label=losses['long_residual_real'][:,2]<losses['long_no_rag'][:,2]
            np.testing.assert_array_equal(f['label'],label)
        policy=read(folder/'policy.json')
        close(policy['quartiles'],np.quantile(features,[.25,.75],axis=0).T.tolist())
        close(policy['medians'],np.median(features,axis=0).tolist())
        close(policy['val_real_gain_nmse'],float(np.mean(losses['long_no_rag'][:,2]-losses['long_residual_real'][:,2])))
        validation[name]=(features,label);del d,preds,losses;gc.collect()
    for name in names:
        others=[n for n in names if n!=name];x=np.concatenate([validation[n][0] for n in others]);y=np.concatenate([validation[n][1] for n in others])
        weights=np.concatenate([np.full(len(validation[n][1]),1/len(validation[n][1])) for n in others])
        models={family:fit_sign_model(x,y,weights,columns) for family,columns in
            [('query_only',QUERY_FEATURES),('case_aware',tuple(range(len(FEATURES))))]}
        close(read(root/name/'sign_models.json'),models)
    for name in names:
        folder=root/name;record=read(folder/'test_evidence.json')
        if sha256(folder/'test_rows.npz')!=record['rows_sha256']:raise AssertionError('Changed row losses')
        d,evidence=load(name,'test');features,preds,losses,error=compute(d,evidence);physical+=evidence['physical_case_pairs']
        with np.load(folder/'test_rows.npz',allow_pickle=False) as f:
            np.testing.assert_allclose(f['features'],features,rtol=1e-12,atol=1e-12)
            for key,value in losses.items():np.testing.assert_allclose(f['loss_'+key],value,rtol=1e-12,atol=1e-10)
            np.testing.assert_allclose(f['real_long'],preds['long_residual_real'],rtol=1e-12,atol=1e-10)
        rebuilt=summarize(d,evidence,features,losses,read(folder/'policy.json'),read(folder/'sign_models.json'))
        close(read(folder/'result.json'),rebuilt)
        close(next(r for r in summary['records'] if r['dataset']==name),rebuilt)
        # Secondary evidence for execution order, not a substitute for hash checks.
        if (root/'frozen.json').stat().st_mtime_ns> (folder/'test_rows.npz').stat().st_mtime_ns:raise AssertionError('TEST predates freeze')
        print('EFFECT VERIFIED',name,flush=True);del d,preds,losses;gc.collect()
    close(read(root/'synthetic.json'),synthetic())
    report=dict(passed=True,datasets=names,physical_case_pairs=physical,
        checked=['Physical histories/targets, all real/random case pairs and temporal boundaries',
            'All VAL feature thresholds, exact zero-training interventions and TEST row metrics',
            'LODO coefficients rebuilt without target-dataset benefit labels',
            'Four derangement losses, sign diagnostics, gate outcomes, groups and intervals',
            'Twenty paired synthetic seeds rebuilt'],
        caveat='Reproduction within the same frozen inputs/software; not a fresh blind experiment')
    write_json(root/'verification.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    print(verify(Path(p.parse_args().root)))
