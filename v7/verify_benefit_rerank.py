"""Rebuild candidate inputs, train-only fitting, guarded policy and raw metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import sha256,write_json
from .benchmark_error_correction import load
from .benchmark_benefit_rerank import corrections
from .trend_rag import ResidualPool,decompose,normalize
from .benefit_rerank import pair_features,fit_ridge,guarded_policy,apply_policy,RIDGES


def verify(root,source):
    root,source=Path(root),Path(source);protocol=json.loads((root/'protocol.json').read_text());audits=[]
    cache_roots={Path(c).name:Path(c) for c in protocol['caches']}
    for path in sorted(root.glob('*/*/run.json')):
        folder=path.parent;name=folder.name;cache=cache_roots[folder.parent.name]/name
        run=json.loads(path.read_text());meta=json.loads((folder/'manifest.json').read_text());old=json.loads((cache/'manifest.json').read_text())
        assert meta['source_manifest_sha256']==sha256(cache/'manifest.json')
        store=Store(source/name/'store');assert store.meta['data_id']==meta['source_data_id']
        libraries={}
        for s in store.series:
            pool=ResidualPool(store,s['sid'],meta['window'],end=s['n'] if meta['memory_scope']=='observed' else None,stride=meta['memory_stride'])
            past=np.stack([store.window(s['sid'],p,96) for p in pool.starts]);errors=None
            if meta['value_mode']=='errors':
                p=cache/f"memory_errors_{s['sid']}.npz";assert sha256(p)==old['memory_files'][str(s['sid'])]
                with np.load(p,allow_pickle=False) as f:errors=f['errors'].copy()
            libraries[s['sid']]=(pool,past,errors)
        tables={};pairs=0
        for split in ['train','validation','test']:
            p=folder/(split+'.npz');assert sha256(p)==meta['splits'][split]
            with np.load(p,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
            original=load(cache,split)
            for k in ['sid','start','base','target','raw_y','std']:np.testing.assert_array_equal(data[k],original[k])
            np.testing.assert_array_equal(data['starts'],original['candidate_starts'])
            for i,(sid,start) in enumerate(zip(data['sid'],data['start'])):
                sid,start=int(sid),int(start);s=store.series[sid];pool,past,old_errors=libraries[sid]
                ids=np.searchsorted(pool.starts,data['starts'][i]);np.testing.assert_array_equal(pool.starts[ids],data['starts'][i])
                assert len(np.unique(data['starts'][i]))==100
                assert np.all(data['starts'][i]+192<=min(start,pool.memory_end))
                raw=store.window(sid,start,96);std=max(float(s['memory_std']),1e-6)
                if meta['value_mode']=='errors':expected=old_errors[ids]
                else:
                    _,r=decompose(raw,meta['window']);_,m,sc=normalize(r,pool.floor)
                    _,pr=decompose(np.r_[raw,data['base'][i]],meta['window'])
                    expected=((pool.future[ids]*sc+m-pr[96:])/std).astype(np.float32)
                np.testing.assert_array_equal(expected,data['values'][i])
                feature=pair_features(raw,data['base'][i],start,data['starts'][i],past[ids],pool.past[ids],expected,
                    original['future_scores'][i],original['history_scores'][i],s['memory_mean'],std,meta['period'],meta['window'])
                np.testing.assert_array_equal(feature,data['features'][i]);pairs+=100
            tables[split]=data
        assert sha256(folder/'ranker.npz')==run['ranker_sha256']
        assert sha256(folder/'policy.json')==run['policy_sha256']
        assert sha256(folder/'predictions.npz')==run['predictions_sha256']
        policy=json.loads((folder/'policy.json').read_text());selection=policy['model_selection'];train=tables['train']
        for fold in selection['folds']:
            assert fold['train_max_episode_end']<=fold['heldout_first_start']
            assert fold['train_queries']==int(np.sum(train['start']+192<=fold['heldout_first_start']))
        assert selection['selected_penalty']==min(RIDGES,key=lambda p:(np.mean(selection['forward_nmse'][str(p)]),-p))
        with np.load(folder/'ranker.npz',allow_pickle=False) as f:state={k:f[k].copy() for k in f.files}
        labels=np.mean(train['values']*train['target'][:,None,:],axis=-1)
        rebuilt=fit_ridge(train['features'].reshape(-1,train['features'].shape[-1]),labels.reshape(-1),selection['selected_penalty'])
        for k in state:np.testing.assert_allclose(state[k],rebuilt[k],atol=1e-8,rtol=1e-8)
        val=tables['validation'];v=corrections(state,val)
        assert guarded_policy(val['target'],v['correction'],val['sid'],val['start'])==policy['utility']
        assert guarded_policy(val['target'],val['prior_correction'],val['sid'],val['start'])==policy['prior']
        test=tables['test'];t=corrections(state,test);delta=apply_policy(policy['utility'],t['correction'],test['sid'])
        with np.load(folder/'predictions.npz',allow_pickle=False) as f:
            np.testing.assert_array_equal(f['selected'],t['selected']);np.testing.assert_array_equal(f['weights'],t['weights'])
            np.testing.assert_array_equal(f['utility_stable'],test['base']+delta*test['std'][:,None])
            for method in run['metrics']:
                error=f[method].astype(float)-test['raw_y'];mse=float(np.mean(error**2));mae=float(np.mean(abs(error)))
                nmse=float(np.mean((error/test['std'][:,None])**2))
                for metric,value in [('mse',mse),('mae',mae),('nmse',nmse)]:assert np.isclose(run['metrics'][method][metric],value,atol=1e-10,rtol=1e-10)
        audits.append(dict(cache=folder.parent.name,dataset=name,candidate_pairs_checked=pairs,overlap_violations=0,
            train_only_model_rebuilt=True,purged_folds_checked=True,guarded_policy_reconstructed=True,raw_metrics_recomputed=True))
        store.close();print('VERIFIED BENEFIT',audits[-1],flush=True)
    write_json(root/'verification.json',dict(complete=True,audits=audits))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    a=p.parse_args();verify(a.root,a.source)
