"""Rebuild the frozen history-length experiment and check its physical inputs."""
import argparse
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import sha256,write_json
from .benchmark_memory_access import read
from .benchmark_history_length import read_npz
from .history_length import (LENGTHS,observed_context,common_eligible,metrics,
    block_metrics,select_lengths,paired_interval,plateau_diagnostic)


def equal(actual,expected):
    if isinstance(expected,dict):
        if actual.keys()!=expected.keys():raise AssertionError('Different fields')
        for k,v in expected.items():equal(actual[k],v)
    elif isinstance(expected,list):
        if len(actual)!=len(expected):raise AssertionError('Different lengths')
        for a,b in zip(actual,expected):equal(a,b)
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
    elif actual!=expected:raise AssertionError((actual,expected))


def verify(root):
    summary=read(root/'summary.json');protocol=read(root/'protocol.json')
    if not summary['complete']:raise AssertionError('Incomplete experiment')
    equal(summary['protocol'],protocol)
    frozen=read(root/'frozen.json');reports=[]
    for name in protocol['datasets']:
        folder=root/name
        if sha256(folder/'policy.json')!=frozen[name]:raise AssertionError('Changed frozen policy')
        rebuilt={};physical=0
        for split in ('validation','test'):
            record=read(folder/f'{split}_cohort.json');path=folder/f'{split}_cohort.npz'
            if sha256(path)!=record['cohort_sha256']:raise AssertionError('Changed cohort')
            d=read_npz(path);store=Store(record['store'])
            try:
                if sha256(Path(record['store'])/'catalog.json')!=record['catalog_sha256']:raise AssertionError('Changed catalog')
                for sid in np.unique(d['sid']):
                    sid=int(sid);series=store.series[sid]
                    if sha256(Path(record['store'])/series['file'])!=record['values_sha256'][str(sid)]:raise AssertionError('Changed series')
                    rows=np.flatnonzero(d['sid']==sid)
                    for i in rows[np.unique([0,len(rows)//2,len(rows)-1])]:
                        t=int(d['start'][i]);boundary=t+96
                        if not common_eligible(series,t):raise AssertionError('Invalid context')
                        values=store.values(sid);x=observed_context(values,t,4096)
                        if not x.flags.owndata:raise AssertionError('Context must own storage')
                        np.testing.assert_array_equal(store.window(sid,boundary,96),d['raw_y'][i])
                        # Poison only the unobserved tail in an independent local slice.
                        local=np.array(values[boundary-4096:boundary+96],copy=True)
                        expected=observed_context(local,4000,4096)
                        local[4096:]=np.nan
                        np.testing.assert_array_equal(observed_context(local,4000,4096),expected)
                        np.testing.assert_array_equal(x,expected);physical+=1
            finally:store.close()
            predictions={96:d['moirai_96']}
            for length in LENGTHS[1:]:
                manifest=read(folder/f'{split}_{length}.json');path=folder/f'{split}_{length}.npz'
                if manifest['cohort_sha256']!=record['cohort_sha256'] or sha256(path)!=manifest['prediction_sha256']:raise AssertionError('Changed prediction')
                p=read_npz(path)
                for key in ('sid','start'):np.testing.assert_array_equal(d[key],p[key])
                if not np.isfinite(p['prediction']).all():raise AssertionError('Nonfinite prediction')
                predictions[length]=p['prediction']
            scores={str(k):metrics(d['raw_y'],p,d['train_std']) for k,p in predictions.items()}
            blocks=block_metrics(d,predictions);saved=read(folder/f'{split}_metrics.json')
            equal(saved['metrics'],scores);equal(saved['blocks'],blocks)
            rebuilt[split]=(d,predictions,scores,blocks,record)
        equal(read(folder/'policy.json'),select_lengths(rebuilt['validation'][2],rebuilt['validation'][3]))
        d,predictions,scores,blocks,record=rebuilt['test'];result=read(folder/'result.json')
        policy=read(folder/'policy.json');selected=predictions[policy['primary_length']];old=d['restored_5plus5']
        equal(result['metrics'],scores)
        equal(result['selected_metrics'],scores[str(policy['primary_length'])])
        equal(result['robust_secondary_metrics'],scores[str(policy['robust_secondary_length'])])
        equal(result['original5plus5_metrics'],metrics(d['raw_y'],old,d['train_std']))
        pairs={'selected_vs96':(predictions[96],selected),'selected_vs_original5plus5':(old,selected),
            'original5plus5_vs_selected':(selected,old),'length4096_vs2016':(predictions[2016],predictions[4096])}
        for key,(base,pred) in pairs.items():equal(result['comparisons'][key],paired_interval(d,base,pred,record['period']))
        equal(result['plateau'],plateau_diagnostic(result['comparisons']['length4096_vs2016']))
        for saved,b in zip(result['blocks'],blocks):
            for key,value in b.items():equal(saved[key],value)
            rows=(d['start']>=b['first_start'])&(d['start']<=b['last_start'])
            o=metrics(d['raw_y'][rows],old[rows],d['train_std'][rows])
            equal(saved['original5plus5_metrics'],o)
            equal(saved['selected_gain_vs96_percent'],100*(1-b['metrics'][str(policy['primary_length'])]['mse']/b['metrics']['96']['mse']))
        for variable in read(folder/'variable_error_contributions.json'):
            rows=d['sid']==variable['sid']
            for key,pred in [('baseline',predictions[96]),('selected',selected),('original',old)]:
                equal(variable[key+'_sse'],float(np.sum((pred[rows].astype(float)-d['raw_y'][rows])**2)))
        equal(next(r for r in summary['records'] if r['dataset']==name),result)
        reports.append(dict(dataset=name,queries=record['queries'],physical_query_checks=physical,
            metrics_rebuilt=True,policies_rebuilt=True,bootstrap_rebuilt=True))
    report=dict(passed=True,datasets=reports,scope='Hashes, physical inputs, future-tail poisoning, VAL selection, TEST metrics and intervals; no fresh model inference')
    write_json(root/'verification.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    print(verify(Path(p.parse_args().root)))
