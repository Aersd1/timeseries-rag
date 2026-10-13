"""Supplement the predeclared mean metric; never remove variables or choose models."""
import argparse
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import write_json


def diagnose(root,source,check_electricity=False):
    root,source=Path(root),Path(source);rows=[];robust=[]
    for path in sorted(root.glob('*/run.json')):
        name=path.parent.name;catalog=json.loads((source/name/'store/catalog.json').read_text(encoding='utf-8'))
        with np.load(path.parent/'predictions.npz',allow_pickle=False) as f:
            for sid in np.unique(f['sid']):
                ids=f['sid']==sid;truth=f['raw_y'][ids];std=f['std'][ids,None]
                row=dict(dataset=name,sid=int(sid),column=catalog['series'][sid]['column'],queries=int(ids.sum()))
                for method in ['moirai_direct','prior_stable','utility_shape','utility_state']:
                    row[method]=float(np.mean(((f[method][ids].astype(float)-truth)/std)**2))
                row['state_minus_direct']=row['utility_state']-row['moirai_direct']
                row['state_minus_shape']=row['utility_state']-row['utility_shape'];rows.append(row)
        group=[x for x in rows if x['dataset']==name];direct=np.array([x['moirai_direct'] for x in group])
        result=dict(dataset=name,variables=len(group),methods={})
        for method in ['moirai_direct','prior_stable','utility_shape','utility_state']:
            loss=np.array([x[method] for x in group])
            result['methods'][method]=dict(median_variable_nmse=float(np.median(loss)),
                improved_variables=int(np.sum(loss<direct-1e-10)),worse_variables=int(np.sum(loss>direct+1e-10)),
                unchanged_variables=int(np.sum(abs(loss-direct)<=1e-10)))
        robust.append(result)
    write_json(root/'per_variable_metrics.json',rows);write_json(root/'robust_diagnostics.json',robust)
    if check_electricity:
        import pandas as pd
        group=[x for x in rows if x['dataset']=='electricity']
        if not group:raise ValueError('No electricity results')
        worst=max(group,key=lambda x:x['moirai_direct']);store=Store(source/'electricity/store')
        try:
            s=store.series[worst['sid']];values=np.array(store.values(s['sid']))
            cfg=json.loads((source/'electricity/config.json').read_text(encoding='utf-8'))
            time_column=cfg['data']['sources'][0]['time_column']
            frame=pd.read_csv(s['source'],usecols=[time_column,s['column']])
            np.testing.assert_array_equal(values,frame[s['column']].to_numpy(dtype=np.float32))
            times=pd.to_datetime(frame[time_column])
            def stats(a):
                return dict(rows=len(a),finite_fraction=float(np.isfinite(a).mean()),mean=float(a.mean()),std=float(a.std()),
                    zero_fraction=float((a==0).mean()),quantiles=np.quantile(a,[0,.25,.5,.75,1]).tolist())
            sections={'memory':(0,s['memory_end']),'train':(s['memory_end'],s['train_end']),
                'validation':(s['train_end'],s['validation_end']),'test':(s['validation_end'],s['n'])}
            write_json(root/'electricity_quality.json',dict(dataset='electricity',column=s['column'],rows=len(values),
                source_column_equals_store=True,duplicate_timestamps=int(times.duplicated().sum()),
                hourly_intervals_valid=bool((times.diff().dropna().dt.total_seconds()==3600).all()),
                memory_std=s['memory_std'],profiles={k:stats(values[a:b].astype(float)) for k,(a,b) in sections.items()},
                interpretation='Inspect scale shift without changing the primary metric; actual cause is unknown.',
                caveat='Worst variable is a post-hoc diagnostic, not an exclusion or model-selection rule.'))
        finally:store.close()
    print('DIAGNOSED STATE',[(r['dataset'],r['variables']) for r in robust],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--check-electricity',action='store_true');a=p.parse_args();diagnose(a.root,a.source,a.check_electricity)
