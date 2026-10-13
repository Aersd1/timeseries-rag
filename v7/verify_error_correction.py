"""Rebuild old errors, history-only selections, adoption and paired metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from .common import sha256,write_json
from .trend_rag import ResidualPool
from .error_correction import retrieve,features,innovations,predict_delta
from .benchmark_error_correction import load,BETAS


def verify(root,source):
    root,source=Path(root),Path(source);summary=json.loads((root/'summary.json').read_text());audits=[]
    for record in summary['results']:
        name=record['dataset'];folder=root/name;meta=json.loads((folder/'manifest.json').read_text())
        store=Store(source/name/'store');assert store.meta['data_id']==meta['source_data_id']
        pools={s['sid']:ResidualPool(store,s['sid'],meta['window'],
            end=s['n'] if meta.get('memory_scope')=='observed' else None,stride=meta.get('memory_stride',1)) for s in store.series}
        errors={}
        for sid,pool in pools.items():
            if meta.get('value_mode')=='innovations':continue
            path=folder/f'memory_errors_{sid}.npz';assert sha256(path)==meta['memory_files'][str(sid)]
            with np.load(path,allow_pickle=False) as d:
                np.testing.assert_array_equal(pool.starts,d['starts'])
                truth=np.stack([store.window(sid,int(p)+96,96) for p in pool.starts])
                expected=(truth-d['baseline'])/max(store.series[sid]['memory_std'],1e-6)
                np.testing.assert_array_equal(expected,d['errors']);errors[sid]=d['errors'].copy()
        checked=0;data={s:load(folder,s) for s in ['train','validation','test']}
        for split,d in data.items():
            for i,(sid,start) in enumerate(zip(d['sid'],d['start'])):
                sid,start=int(sid),int(start);pool=pools[sid];s=store.series[sid]
                raw=store.window(sid,start,192)
                np.testing.assert_array_equal(raw[:96],d['raw_x'][i]);np.testing.assert_array_equal(raw[96:],d['raw_y'][i])
                assert np.all(d['candidate_starts'][i]+192<=min(start,pool.memory_end))
                q,m=retrieve(raw[:96],d['raw_quantiles'][i],pool,start,meta['window'])
                for k in ['selected_starts','candidate_starts','weights','future_scores','history_scores']:
                    np.testing.assert_allclose(m[k],d[k][i],atol=1e-6,rtol=1e-6)
                if meta.get('value_mode')=='innovations':
                    e,global_value=innovations(raw[:96],d['raw_quantiles'][i],pool,m,d['std'][i],meta['window'],start)
                else:
                    ids=np.searchsorted(pool.starts,m['selected_starts']);e=errors[sid][ids]
                    global_value=errors[sid][pool.starts+192<=min(start,pool.memory_end)].mean(0)
                np.testing.assert_allclose(global_value,d['global_error'][i],atol=1e-6,rtol=1e-6)
                np.testing.assert_array_equal(e,d['errors'][i]);np.testing.assert_array_equal(m['examples'][:,:96],d['keys'][i])
                f=features(raw[:96],d['raw_quantiles'][i],q,m,e,d['std'][i])
                np.testing.assert_array_equal(f,d['features'][i]);checked+=100
        state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        assert sha256(folder/'best.pt')==record['checkpoint_sha256']
        assert sha256(folder/'predictions.npz')==record['predictions_sha256']
        train,val,test=data['train'],data['validation'],data['test']
        np.testing.assert_array_equal(state['mean'].numpy(),train['features'].mean(0))
        np.testing.assert_array_equal(state['scale'].numpy(),np.maximum(train['features'].std(0),.1))
        weighted=(val['weights'][...,None]*val['errors']).sum(1)
        fixed={str(b):float(np.mean((b*weighted-val['target'])**2)) for b in BETAS}
        assert state['beta']==min(BETAS,key=lambda b:(fixed[str(b)],b))
        learned=dict(state,adopted='learned')
        learned_validation=float(np.mean((predict_delta(learned,val)-val['target'])**2))
        assert np.isclose(learned_validation,record['validation_learned'],rtol=1e-5)
        assert state['adopted']==('learned' if learned_validation<fixed[str(state['beta'])] else 'fixed')
        history=json.loads((folder/'history.json').read_text());assert state['epoch']==min(history,key=lambda x:x['validation_nmse'])['epoch']
        delta=predict_delta(state,test);forecast=test['base']+delta*test['std'][:,None]
        direct_error=(test['base'].astype(float)-test['raw_y'])/test['std'][:,None]
        rag_error=(forecast.astype(float)-test['raw_y'])/test['std'][:,None]
        with np.load(folder/'predictions.npz',allow_pickle=False) as pred:
            np.testing.assert_allclose(pred['rag_adopted'],forecast,atol=1e-6,rtol=1e-6)
            for method in pred.files:
                score=float(np.mean(((pred[method].astype(float)-test['raw_y'])/test['std'][:,None])**2))
                assert np.isclose(score,record['test_nmse'][method],atol=1e-6,rtol=1e-5)
        blocks=np.unique(test['start']);losses=np.array([[np.mean(d[test['start']==b]**2) for d in [direct_error,rag_error]] for b in blocks])
        rng=np.random.default_rng(20261013);sample=losses[rng.integers(0,len(blocks),size=(2000,len(blocks)))].mean(1)
        interval=np.quantile(100*(1-sample[:,1]/sample[:,0]),[.025,.975]).tolist()
        audits.append(dict(dataset=name,candidate_pairs_checked=checked,overlap_violations=0,
            raw_metrics_recomputed=True,adoption_reconstructed=True,baseline_reproduction_max_abs_difference=meta['baseline_max_abs_difference'],
            paired_temporal_blocks=len(blocks),rag_gain_bootstrap_95_interval_percent=interval))
        store.close();print('VERIFIED ERROR TRANSFER',name,audits[-1],flush=True)
    write_json(root/'verification.json',dict(complete=True,audits=audits))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    args=p.parse_args();torch.set_num_threads(4);verify(args.root,args.source)
