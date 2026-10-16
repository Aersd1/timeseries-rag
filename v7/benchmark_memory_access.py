"""Quick, predeclared sparse-cohort memory access comparison; no training."""
import argparse,json,time,gc
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import sha256,write_json
from .current_forecast import DATASETS,RUN_ROOT,PRIMARY
from .memory_access import CONTEXT,align_pack,select_clock,history_controls

SOURCE=Path('v7/runs/unified_residual_5plus5_96_20261015')
ARMS=('moirai_96','moirai_512','moirai_2016','packed_real','packed_random',
      'packed_periodic','packed_recent','packed_univariate','restored_5plus5')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare(name,out):
    begin=time.perf_counter();source=SOURCE/name;meta=read(source/'preparation.json')
    path=source/'test.npz';info=read(source/'test_info.json')
    if sha256(path)!=info['sha256']:raise ValueError('Changed frozen test inputs')
    with np.load(path,allow_pickle=False) as f:
        sid=f['sid'];start=f['start'];origins=np.unique(start)
        positions=np.unique(np.rint(np.linspace(0,len(origins)-1,3)).astype(int));chosen=origins[positions]
        ids=np.flatnonzero(np.isin(start,chosen))
        d={k:f[k][ids].copy() for k in ['sid','start','raw_x','raw_y','raw_quantiles',
            'selected_starts','case_past','case_future','random_selected_starts','random_case_past','random_case_future']}
    old_path=RUN_ROOT/name/'predictions.npz';old_run=read(RUN_ROOT/name/'run.json')
    if sha256(old_path)!=old_run['predictions_sha256']:raise ValueError('Changed incumbent')
    with np.load(old_path,allow_pickle=False) as f:
        for k in ('sid','start','raw_y','selected_starts'):np.testing.assert_array_equal(d[k],f[k][ids])
        old=f[PRIMARY][ids].copy()
    stats=read(source/'stats.json')
    if sha256(source/'stats.json')!=meta['stats_sha256']:raise ValueError('Changed statistics')
    store=Store(meta['store']);kept=[];records=[];contexts={k:[] for k in ARMS if k not in ('moirai_96','restored_5plus5')}
    selected={k:[] for k in ('real','random','periodic','recent','univariate')}
    try:
        for sid_value in np.unique(d['sid']):
            sid_value=int(sid_value);s=store.series[sid_value];values=store.values(sid_value)
            limit=int(meta['limits'][str(sid_value)])
            starts=np.array([p for a,b in s['runs'] for p in range(a,min(b,limit)-192+1)],dtype=np.int64)
            for i in np.flatnonzero(d['sid']==sid_value):
                t=int(d['start'][i]);end=t+96;left=end-CONTEXT
                if left<0 or not any(a<=left and end+96<=b for a,b in s['runs']):continue
                # Own this storage: Store.close() invalidates its memory-mapped views.
                long=np.array(values[left:end],dtype=np.float32,copy=True)
                if not np.isfinite(long).all():continue
                np.testing.assert_array_equal(long[-96:],d['raw_x'][i])
                np.testing.assert_array_equal(values[end:end+96],d['raw_y'][i])
                choices={'real':d['selected_starts'][i], 'random':d['random_selected_starts'][i],
                         'periodic':select_clock(starts,t,limit,meta['period'],'periodic'),
                         'recent':select_clock(starts,t,limit,meta['period'],'recent'),
                         'univariate':history_controls(values,starts,d['raw_x'][i],t,limit)}
                for arm,case_starts in choices.items():
                    if len(np.unique(case_starts))!=10 or np.any(case_starts+192>min(t,limit)):
                        raise ValueError('Invalid historical case boundary')
                    past=np.asarray(values[case_starts[:,None]+np.arange(96)])
                    future=np.asarray(values[case_starts[:,None]+96+np.arange(96)])
                    if arm in ('real','random'):
                        prefix='' if arm=='real' else 'random_'
                        np.testing.assert_array_equal(past,d[prefix+'case_past'][i])
                        np.testing.assert_array_equal(future,d[prefix+'case_future'][i])
                    contexts['packed_'+arm].append(align_pack(d['raw_x'][i],past,future,case_starts))
                    selected[arm].append(case_starts)
                contexts['moirai_512'].append(long[-512:]);contexts['moirai_2016'].append(long)
                kept.append(int(i))
                if len(kept)%250==0:print('MEMORY PREP',name,len(kept),flush=True)
    finally:store.close()
    kept=np.asarray(kept,dtype=int)
    if not len(kept):raise ValueError('No common finite cohort')
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'inputs.npz',**{k:np.stack(v) for k,v in contexts.items()},
                        **{'selected_'+k:np.stack(v) for k,v in selected.items()},
                        sid=d['sid'][kept],start=d['start'][kept],raw_y=d['raw_y'][kept],
                        moirai_96=d['raw_quantiles'][kept,4],restored_5plus5=old[kept],
                        train_std=np.asarray(stats['std'])[d['sid'][kept]])
    write_json(out/'cohort.json',dict(dataset=name,chosen_origins=chosen.tolist(),
        source_rows=ids.tolist(),kept_rows=kept.tolist(),queries=len(kept),filtered=len(ids)-len(kept),
        variables=len(np.unique(d['sid'][kept])),source_data_id=meta['source_data_id'],
        store=meta['store'],period=meta['period'],limits=meta['limits'],moirai=read(Path(meta['source_cache'])/'manifest.json')['moirai'],
        source_sha256=info['sha256'],inputs_sha256=sha256(out/'inputs.npz'),preparation_seconds=time.perf_counter()-begin,
        note='First/middle/last existing TEST origins, all variables; common finite2016 cohort; no target-based selection'))
    print('MEMORY COHORT',name,len(kept),'filtered',len(ids)-len(kept),round(time.perf_counter()-begin,1),flush=True)


def score(y,p,sd):
    e=np.asarray(p,dtype=float)-y
    return dict(mse=float(np.mean(e**2)),mae=float(np.mean(abs(e))),train_nmse=float(np.mean((e/sd[:,None])**2)))


def run(root,names,batch_size):
    import torch
    from .moirai import MoiraiAdapter
    meta=read(root/names[0]/'cohort.json');settings=dict(meta['moirai'],local_files_only=True)
    model=MoiraiAdapter.pretrained(settings,'cuda');records=[]
    for name in names:
        folder=root/name;c=read(folder/'cohort.json')
        if sha256(folder/'inputs.npz')!=c['inputs_sha256']:raise ValueError('Changed frozen cohort')
        with np.load(folder/'inputs.npz',allow_pickle=False) as f:
            truth=f['raw_y'];sd=f['train_std'];predictions={k:f[k].copy() for k in ('moirai_96','restored_5plus5')}
            times={}
            for arm in ARMS:
                if arm in predictions:continue
                x=f[arm];parts=[];torch.cuda.synchronize();begin=time.perf_counter()
                for j in range(0,len(x),batch_size):
                    parts.append(model.forecast(x[j:j+batch_size],96)[:,4].cpu().numpy())
                torch.cuda.synchronize();times[arm]=time.perf_counter()-begin
                predictions[arm]=np.concatenate(parts)
                print('MEMORY FORECAST',name,arm,len(x),round(times[arm],2),flush=True)
            # Real foundation smoke check, without replacing the pinned cached96 baseline.
            probe=model.forecast(f['moirai_2016'][:1,-96:],96)[:,4].cpu().numpy()
            np.testing.assert_allclose(probe[0],predictions['moirai_96'][0],rtol=2e-5,atol=2e-4)
            metrics={k:score(truth,p,sd) for k,p in predictions.items()}
            np.savez_compressed(folder/'predictions.npz',**predictions,sid=f['sid'],start=f['start'])
        result=dict(dataset=name,queries=c['queries'],variables=c['variables'],filtered=c['filtered'],
                    metrics=metrics,gpu_seconds=times,
                    gain_vs_moirai96_percent={k:100*(1-v['mse']/metrics['moirai_96']['mse']) for k,v in metrics.items()})
        write_json(folder/'result.json',result);records.append(result)
        print('MEMORY RESULT',name,result['gain_vs_moirai96_percent'],flush=True)
        gc.collect()
    write_json(root/'summary.json',dict(complete=True,arms=list(ARMS),records=records,
        caveats=['Only three fixed origins per dataset; sparse exploratory screen, not a full benchmark',
                 'Original retrieval can use sibling-channel histories; univariate control does not',
                 'Packing is naive concatenation with history-only affine alignment, not a specially trained demonstration interface',
                 'Same2016 numeric context length does not equal total retrieval/training compute',
                 'Previously explored TEST; no threshold/model selection or default replacement']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    p.add_argument('--batch-size',type=int,default=32);a=p.parse_args();root=Path(a.output)
    if root.exists():raise FileExistsError('Choose a fresh quick experiment')
    root.mkdir(parents=True)
    write_json(root/'protocol.json',dict(datasets=a.datasets,arms=list(ARMS),origins_per_dataset=3,
        main_comparison='packed_real versus continuous2016, packed_random, packed_periodic, packed_recent and same-channel history selection',
        context_points=CONTEXT,forecast_horizon=96,training=False,default_unchanged=True))
    for name in a.datasets:prepare(name,root/name)
    write_json(root/'frozen.json',{name:sha256(root/name/'cohort.json') for name in a.datasets})
    run(root,a.datasets,a.batch_size)


if __name__=='__main__':main()
