"""Full existing cohort sweep. All VAL selections freeze before any new TEST forecast."""
import argparse,gc,time
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import sha256,write_json
from .current_forecast import DATASETS,RUN_ROOT,PRIMARY
from .benchmark_memory_access import read
from .history_length import (LENGTHS,observed_context,common_eligible,metrics,block_metrics,
                              select_lengths,paired_interval,plateau_diagnostic)

SOURCE=Path('v7/runs/unified_residual_5plus5_96_20261015')


def read_npz(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def prepare(name,split,folder):
    folder.mkdir(parents=True,exist_ok=True);record_path=folder/f'{split}_cohort.json'
    source=SOURCE/name;meta=read(source/'preparation.json');info=read(source/f'{split}_info.json')
    if record_path.exists():
        old=read(record_path)
        if old['source_sha256']!=info['sha256'] or sha256(folder/f'{split}_cohort.npz')!=old['cohort_sha256']:
            raise ValueError('Changed prepared cohort')
        return old
    path=source/f'{split}.npz'
    if sha256(path)!=info['sha256']:raise ValueError('Changed source '+split)
    if sha256(source/'stats.json')!=meta['stats_sha256']:raise ValueError('Changed TRAIN metrics scale')
    with np.load(path,allow_pickle=False) as f:
        data={k:f[k].copy() for k in ('sid','start','raw_x','raw_y','raw_quantiles')}
    stats=read(source/'stats.json');store=Store(meta['store']);ids=[];value_hashes={}
    try:
        if store.meta['data_id']!=meta['source_data_id']:raise ValueError('Wrong physical Store')
        for sid in np.unique(data['sid']):
            sid=int(sid);series=store.series[sid]
            h=sha256(Path(meta['store'])/series['file'])
            if h!=series['values_sha256']:raise ValueError('Changed raw series')
            value_hashes[str(sid)]=h
            for row in np.flatnonzero(data['sid']==sid):
                t=int(data['start'][row])
                if not common_eligible(series,t):continue
                x=observed_context(store.values(sid),t,4096)
                np.testing.assert_array_equal(x[-96:],data['raw_x'][row])
                np.testing.assert_array_equal(store.window(sid,t+96,96),data['raw_y'][row])
                if split=='validation' and (t<series['train_end'] or t+192>series['validation_end']):raise ValueError('Invalid VAL target')
                if split=='test' and (t<series['validation_end'] or t+192>series['n']):raise ValueError('Invalid TEST target')
                ids.append(int(row))
    finally:store.close()
    ids=np.sort(np.asarray(ids,dtype=int))
    if not len(ids):raise ValueError('No common4096 cohort')
    saved={k:data[k][ids] for k in ('sid','start','raw_y')}
    saved.update(source_rows=ids,moirai_96=data['raw_quantiles'][ids,4],
                 train_std=np.maximum(np.asarray(stats['std'])[data['sid'][ids]],1e-6))
    if split=='test':
        original=RUN_ROOT/name;run=read(original/'run.json');p=original/'predictions.npz'
        if sha256(p)!=run['predictions_sha256']:raise ValueError('Changed original5+5 result')
        with np.load(p,allow_pickle=False) as f:
            for k in ('sid','start','raw_y'):np.testing.assert_array_equal(f[k][ids],saved[k])
            saved['restored_5plus5']=f[PRIMARY][ids].copy()
    np.savez_compressed(folder/f'{split}_cohort.npz',**saved)
    manifest=read(Path(meta['source_cache'])/'manifest.json')
    record=dict(dataset=name,split=split,source_sha256=info['sha256'],store=meta['store'],
        data_id=meta['source_data_id'],catalog_sha256=sha256(Path(meta['store'])/'catalog.json'),
        values_sha256=value_hashes,period=meta['period'],moirai=manifest['moirai'],
        total_source_queries=len(data['sid']),queries=len(ids),filtered=len(data['sid'])-len(ids),
        origins=len(np.unique(saved['start'])),variables=len(np.unique(saved['sid'])),
        cohort_sha256=sha256(folder/f'{split}_cohort.npz'),
        rule='All existing frozen source queries with common finite4096 observed history; same rows for every length; no imputation')
    write_json(record_path,record);print('LENGTH COHORT',name,split,record['queries'],'origins',record['origins'],'filtered',record['filtered'],flush=True)
    return record


def predict_split(model,name,split,folder,batch_size):
    import torch
    record=read(folder/f'{split}_cohort.json');path=folder/f'{split}_cohort.npz'
    if sha256(path)!=record['cohort_sha256']:raise ValueError('Changed cohort rows')
    if sha256(Path(record['store'])/'catalog.json')!=record['catalog_sha256']:raise ValueError('Changed Store metadata')
    data=read_npz(path);predictions={96:data['moirai_96']};timings={};store=Store(record['store'])
    try:
        for length in LENGTHS[1:]:
            output=folder/f'{split}_{length}.npz';manifest_path=folder/f'{split}_{length}.json'
            if manifest_path.exists():
                m=read(manifest_path)
                if m['cohort_sha256']!=record['cohort_sha256'] or sha256(output)!=m['prediction_sha256']:raise ValueError('Changed saved length forecast')
                predictions[length]=read_npz(output)['prediction'];timings[str(length)]=m['gpu_seconds']
                print('LENGTH REUSE',name,split,length,flush=True);continue
            begin=time.perf_counter();parts=[];last_progress=begin
            for j in range(0,len(data['sid']),batch_size):
                rows=range(j,min(j+batch_size,len(data['sid'])))
                x=np.stack([observed_context(store.values(int(data['sid'][i])),int(data['start'][i]),length) for i in rows])
                p=model.forecast(x,96)[:,4].cpu().numpy();parts.append(p)
                if time.perf_counter()-last_progress>=30:
                    print('LENGTH PROGRESS',name,split,length,min(j+batch_size,len(data['sid'])),'/',len(data['sid']),flush=True)
                    last_progress=time.perf_counter()
            torch.cuda.synchronize();elapsed=time.perf_counter()-begin
            pred=np.concatenate(parts);predictions[length]=pred;timings[str(length)]=elapsed
            np.savez_compressed(output,prediction=pred,sid=data['sid'],start=data['start'])
            write_json(manifest_path,dict(length=length,horizon=96,cohort_sha256=record['cohort_sha256'],
                prediction_sha256=sha256(output),gpu_seconds=elapsed,batch_size=batch_size,model_identity=model.identity))
            print('LENGTH DONE',name,split,length,round(elapsed,1),flush=True)
        smoke_ids=np.unique([0,len(data['sid'])-1])
        if name=='electricity':smoke_ids=np.unique(np.r_[smoke_ids,np.flatnonzero(data['sid']==313)[0]])
        smoke=np.stack([observed_context(store.values(int(data['sid'][i])),int(data['start'][i]),96) for i in smoke_ids])
        actual=model.forecast(smoke,96)[:,4].cpu().numpy()
        np.testing.assert_allclose(actual,data['moirai_96'][smoke_ids],rtol=2e-5,atol=2e-4)
    finally:store.close()
    scores={str(k):metrics(data['raw_y'],v,data['train_std']) for k,v in predictions.items()}
    blocks=block_metrics(data,predictions)
    write_json(folder/f'{split}_metrics.json',dict(metrics=scores,blocks=blocks,timings=timings,
        cached96_smoke_queries=len(smoke_ids),cached96_max_absolute_error=float(np.max(abs(actual-data['moirai_96'][smoke_ids])))))
    return data,predictions,scores,blocks,record


def evaluate_test(name,folder,data,predictions,scores,blocks,record):
    policy=read(folder/'policy.json');length=int(policy['primary_length']);robust=int(policy['robust_secondary_length'])
    selected=predictions[length];old=data['restored_5plus5']
    comparisons=dict(selected_vs96=paired_interval(data,predictions[96],selected,record['period']),
        selected_vs_original5plus5=paired_interval(data,old,selected,record['period']),
        original5plus5_vs_selected=paired_interval(data,selected,old,record['period']),
        length4096_vs2016=paired_interval(data,predictions[2016],predictions[4096],record['period']))
    block_results=[]
    for b in blocks:
        ids=np.flatnonzero((data['start']>=b['first_start'])&(data['start']<=b['last_start']))
        baseline=b['metrics']['96']['mse'];o=metrics(data['raw_y'][ids],old[ids],data['train_std'][ids])
        block_results.append(dict(b,selected_length=length,
            selected_gain_vs96_percent=100*(1-b['metrics'][str(length)]['mse']/baseline),
            original5plus5_metrics=o,
            original_gain_vs96_percent=100*(1-o['mse']/baseline)))
    # Attribute raw error to variables without excluding large-amplitude columns.
    by_variable=[]
    for sid in np.unique(data['sid']):
        ids=data['sid']==sid
        by_variable.append(dict(sid=int(sid),queries=int(ids.sum()),
            baseline_sse=float(np.sum((predictions[96][ids].astype(float)-data['raw_y'][ids])**2)),
            selected_sse=float(np.sum((selected[ids].astype(float)-data['raw_y'][ids])**2)),
            original_sse=float(np.sum((old[ids].astype(float)-data['raw_y'][ids])**2))))
    result=dict(dataset=name,queries=record['queries'],origins=record['origins'],variables=record['variables'],filtered=record['filtered'],
        selected_length=length,robust_secondary_length=robust,metrics=scores,
        selected_metrics=scores[str(length)],robust_secondary_metrics=scores[str(robust)],
        original5plus5_metrics=metrics(data['raw_y'],old,data['train_std']),
        comparisons=comparisons,plateau=plateau_diagnostic(comparisons['length4096_vs2016']),blocks=block_results,
        caveat='Full existing sampled cohorts, not every sliding window; reused TEST; nominal weekly intervals without repeated-exploration correction')
    write_json(folder/'variable_error_contributions.json',by_variable);write_json(folder/'result.json',result)
    print('LENGTH RESULT',name,'VAL_LENGTH',length,'TEST_GAIN',round(comparisons['selected_vs96']['gain_percent'],3),result['plateau']['classification'],flush=True)
    return result


def main():
    import torch
    from .moirai import MoiraiAdapter
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--resume',action='store_true');a=p.parse_args()
    root=Path(a.output);protocol=dict(datasets=a.datasets,lengths=list(LENGTHS),horizon=96,
        primary='Pooled VAL raw-MSE selected length, frozen for TEST',
        secondary='Worst four-block VAL relative-MSE length; diagnostic only',
        test_blocks=4,plateau_margin_percent=1.,training=False,default_unchanged=True,
        cohort='All existing frozen validation/test query rows eligible for finite4096 history, not just the previous3 origins',
        caveats=['Previously explored sampled TEST; not a fresh blind benchmark',
                 'History-length sweep alone cannot prove presence or absence of all information gaps',
                 'Does not yet test correlated variables, external covariates, online calibration or dense every-step forecasts'])
    if root.exists():
        if not a.resume or read(root/'protocol.json')!=protocol:raise ValueError('Use matching --resume or a fresh root')
    else:root.mkdir(parents=True);write_json(root/'protocol.json',protocol)
    for name in a.datasets:prepare(name,'validation',root/name)
    settings=dict(read(root/a.datasets[0]/'validation_cohort.json')['moirai'],local_files_only=True)
    print('LENGTH MODEL LOADING',flush=True);model=MoiraiAdapter.pretrained(settings,'cuda');print('LENGTH MODEL READY',flush=True)
    for name in a.datasets:
        folder=root/name;d,pred,scores,blocks,meta=predict_split(model,name,'validation',folder,a.batch_size)
        chosen=select_lengths(scores,blocks)
        if (folder/'policy.json').exists():
            if read(folder/'policy.json')!=chosen:raise ValueError('Frozen VAL selection changed')
        else:write_json(folder/'policy.json',chosen)
        print('LENGTH POLICY',name,chosen['primary_length'],chosen['robust_secondary_length'],flush=True)
        del d,pred;gc.collect()
    frozen={name:sha256(root/name/'policy.json') for name in a.datasets}
    if (root/'frozen.json').exists():
        if read(root/'frozen.json')!=frozen:raise ValueError('All-dataset policy freeze changed')
    else:write_json(root/'frozen.json',frozen)
    records=[]
    # No new TEST cohort or prediction is opened above this boundary.
    for name in a.datasets:
        if sha256(root/name/'policy.json')!=frozen[name]:raise ValueError('Changed selection before TEST')
        prepare(name,'test',root/name)
        d,pred,scores,blocks,meta=predict_split(model,name,'test',root/name,a.batch_size)
        records.append(evaluate_test(name,root/name,d,pred,scores,blocks,meta));del d,pred;gc.collect()
        write_json(root/'progress.json',dict(complete=False,finished_datasets=[r['dataset'] for r in records]))
    write_json(root/'summary.json',dict(complete=True,records=records,protocol=protocol))


if __name__=='__main__':main()
