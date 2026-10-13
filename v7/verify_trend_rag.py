"""Reconstruct cached signals, chronological exclusions and paired raw metrics."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import write_json,sha256
from .benchmark_trend_rag import load_split
from .trend_rag import ResidualPool,normalize


def reference(values,window):
    x=np.asarray(values,dtype=np.float64)
    kernel=np.ones(window)/window
    trend=np.convolve(np.r_[np.repeat(x[0],window-1),x],kernel,mode='valid')
    return trend.astype(np.float32),(x-trend).astype(np.float32)


def verify(root,source):
    root,source=Path(root),Path(source)
    summary=json.loads((root/'summary.json').read_text()); audits=[]
    with (root/'query_metrics.csv').open(encoding='utf-8') as f:
        metrics={(r['dataset'],int(r['query']),r['method']):r for r in csv.DictReader(f)}
    for row in summary['results']:
        name=row['dataset'];folder=root/name
        manifest=json.loads((folder/'manifest.json').read_text());store=Store(source/name/'store')
        if store.meta['data_id']!=manifest['source_data_id']: raise AssertionError('Data identity changed')
        pools={s['sid']:ResidualPool(store,s['sid'],manifest['window']) for s in store.series}
        candidate_count=0;query_count=0
        for split in ['train','validation','test']:
            data=load_split(folder,split)
            for i,(sid,start) in enumerate(zip(data['sid'],data['start'])):
                sid,start=int(sid),int(start);s=store.series[sid];pool=pools[sid]
                lo,hi={'train':(s['memory_end'],s['train_end']),'validation':(s['train_end'],s['validation_end']),
                    'test':(s['validation_end'],s['n'])}[split]
                assert lo<=start and start+192<=hi
                raw=store.window(sid,start,192);t,r=reference(raw,manifest['window'])
                np.testing.assert_array_equal(data['raw_x'][i],raw[:96]);np.testing.assert_array_equal(data['raw_y'][i],raw[96:])
                np.testing.assert_allclose(data['trend'][i],t[:96],atol=1e-5,rtol=1e-6)
                np.testing.assert_allclose(data['residual'][i],r[:96],atol=1e-5,rtol=1e-6)
                q,m,sc=normalize(r[:96],pool.floor)
                np.testing.assert_allclose(data['x'][i],q,atol=2e-5,rtol=2e-5)
                np.testing.assert_allclose(data['y'][i],(r[96:]-m)/sc,atol=2e-5,rtol=2e-5)
                assert len(np.unique(data['candidate_starts'][i]))==100
                assert np.all(data['candidate_starts'][i]+192<=min(start,s['memory_end']))
                # Rebuild retrieved old residual histories and continuations from raw values.
                ids=np.searchsorted(pool.starts,data['selected_starts'][i]);assert np.array_equal(pool.starts[ids],data['selected_starts'][i])
                expected=np.concatenate([pool.past[ids],pool.future[ids]],axis=1)
                np.testing.assert_array_equal(expected,data['examples'][i])
                rebuilt=pool.retrieve(data['x'][i],start,data['residual_quantiles'][i],manifest['quantile_levels'])
                for key in ['candidate_starts','selected_starts','weights','history_scores','future_scores']:
                    np.testing.assert_allclose(rebuilt[key],data[key][i],atol=1e-6,rtol=1e-6)
                candidate_count+=100;query_count+=1
            if split!='train':
                for sid in np.unique(data['sid']):
                    starts=np.sort(data['start'][data['sid']==sid]);assert np.all(np.diff(starts)>=192)
        test=load_split(folder,'test')
        with np.load(folder/'plain/predictions.npz',allow_pickle=False) as plain,np.load(folder/'rag/predictions.npz',allow_pickle=False) as rag:
            predictions=dict(moirai_direct=test['direct'],decomposed_frozen=test['frozen'],
                decomposed_plain_ft=plain['prediction'],decomposed_rag=rag['prediction'],rag_memory_disabled=rag['memory_disabled'])
            variances=np.array([max(store.series[int(sid)]['memory_std'],1e-6)**2 for sid in test['sid']])
            recomputed={}
            for method,pred in predictions.items():
                err=pred.astype(float)-test['raw_y'].astype(float);mse=np.mean(err**2,axis=1);mae=np.mean(abs(err),axis=1)
                for i in range(len(mse)):
                    record=metrics[(name,i,method)]
                    for key,value in [('mse',mse[i]),('mae',mae[i]),('nmse',mse[i]/variances[i])]:
                        assert np.isclose(float(record[key]),value,rtol=1e-10,atol=1e-10)
                recomputed[method]=mse/variances
                assert np.isclose(np.mean(recomputed[method]),row[method]['nmse'],rtol=1e-10)
            # Bootstrap synchronized temporal blocks, not seven correlated variables independently.
            blocks=np.unique(test['start']); losses=np.array([[recomputed[m][test['start']==b].mean()
                for m in ['moirai_direct','decomposed_rag']] for b in blocks])
            rng=np.random.default_rng(20261013);samples=losses[rng.integers(0,len(blocks),size=(2000,len(blocks)))].mean(axis=1)
            gains=100*(1-samples[:,1]/samples[:,0]);ci=np.quantile(gains,[.025,.975]).tolist()
        for variant in ['plain','rag']:
            run=json.loads((folder/variant/'run.json').read_text());history=json.loads((folder/variant/'history.json').read_text())
            assert sha256(folder/variant/'best.pt')==run['checkpoint_sha256']
            assert sha256(folder/variant/'predictions.npz')==run['predictions_sha256']
            assert run['best_epoch']==min(history,key=lambda h:h['validation_nmse'])['epoch']
        audits.append(dict(dataset=name,queries_checked=query_count,candidate_pairs_checked=candidate_count,
            overlap_violations=0,reconstructed_raw_signals=True,metric_rows_checked=len(test['x'])*5,
            paired_temporal_blocks=len(blocks),rag_gain_bootstrap_95_interval_percent=ci))
        store.close();print('VERIFIED',name,audits[-1],flush=True)
    result=dict(complete=True,audits=audits)
    write_json(root/'verification.json',result)
    from .report_trend_rag import report
    report(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    args=p.parse_args();verify(args.root,args.source)
