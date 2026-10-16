"""Fast matched local adaptation on the frozen original 5+5 cohorts."""
import argparse,json,gc,time
from pathlib import Path
import numpy as np
from .common import sha256,write_json
from .current_forecast import DATASETS,RUN_ROOT,PRIMARY
from .local_pair_adaptation import train_global,forecasts

SOURCE=Path('v7/runs/unified_residual_5plus5_96_20261015')
PENALTIES=(1.,10.,100.)
BETAS=(.1,.25,.5,1.)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def load(name,split):
    folder=SOURCE/name;meta=read(folder/'preparation.json');info=read(folder/f'{split}_info.json')
    path=folder/f'{split}.npz'
    if sha256(path)!=info['sha256']:raise ValueError('Changed prepared '+split)
    if sha256(folder/'stats.json')!=meta['stats_sha256']:raise ValueError('Changed TRAIN statistics')
    keys=['raw_x','raw_y','raw_quantiles','sid','start']
    if split!='train':keys += ['selected_starts','case_past','case_future','case_baseline',
                              'random_selected_starts','random_case_past','random_case_future','random_case_baseline']
    with np.load(path,allow_pickle=False) as f:d={k:f[k].copy() for k in keys}
    if split!='train':
        limits=np.array([meta['limits'][str(int(s))] for s in d['sid']])
        for k in ('selected_starts','random_selected_starts'):
            if np.any(d[k]+192>np.minimum(limits,d['start'])[:,None]):raise ValueError('Invalid case time boundary')
    return d,read(folder/'stats.json'),meta


def metric(y,p,sd):
    error=np.asarray(p,dtype=float)-y
    return dict(mse=float(np.mean(error**2)),mae=float(np.mean(abs(error))),nmse=float(np.mean((error/sd[:,None])**2)))


def good(a,b):return a['mse']<b['mse']*(1-1e-5) and a['mae']<=b['mae']*(1+1e-10)


def select(data, candidates, reference, stats, cal, confirm):
    sd=np.maximum(np.asarray(stats['std'])[data['sid']],1e-6);y=data['raw_y'];m=data['raw_quantiles'][:,4]
    def metrics(p,ids):return metric(y[ids],p[ids],sd[ids])
    ref=metrics(reference,cal);best=[]
    for key,p in candidates.items():
        score=metrics(p,cal)
        if good(score,ref):best.append((score['mse'],key))
    if not best:return dict(selected='reference',proposed=None,confirmed=False)
    _,key=min(best);score=metrics(candidates[key],confirm);held=metrics(reference,confirm)
    ok=good(score,held)
    return dict(selected=key if ok else 'reference',proposed=key,confirmed=bool(ok),confirmation=score,reference=held)


def candidates(outputs,gates,baseline,arm):
    result={}
    for penalty in (PENALTIES if arm not in ['mean'] else (0,)):
        key=f'real_{penalty}' if arm=='checked' else f'{arm}_{penalty}'
        raw=outputs['mean_transfer'] if arm=='mean' else outputs[key]
        for beta in BETAS:
            value=baseline+beta*(raw-baseline)
            if arm=='checked':value=np.where(gates[str(penalty)][:,None],value,baseline)
            result[f'{penalty}:{beta}']=value
    return result


def calibrate(name,out):
    start=time.perf_counter();train,stats,meta=load(name,'train')
    if np.any(train['start']+192>np.asarray(stats['train_end'])[train['sid']]):raise ValueError('Unfinished TRAIN query')
    model=train_global(train,stats,meta['period']);del train;gc.collect()
    d,_,_=load(name,'validation');origins=np.unique(d['start']);chunks=np.array_split(origins,3)
    # First two chronological VAL blocks choose once; purge complete192 windows
    # before the independent final-block confirmation history begins.
    cal=np.flatnonzero((d['start']<chunks[2][0])&(d['start']+192<=chunks[2][0]))
    catalog=read(Path(meta['store'])/'catalog.json');test_start=min(s['validation_end'] for s in catalog['series'])
    confirm=np.flatnonzero(np.isin(d['start'],chunks[2])&(d['start']+192<=test_start))
    if not len(cal) or not len(confirm):raise ValueError('Empty confirmation split')
    outputs,gates=forecasts(model,d,stats,meta['period']);base=d['raw_quantiles'][:,4]
    query={str(b):base+b*(outputs['global_raw']-base) for b in BETAS}
    policies={'global':select(d,query,base,stats,cal,confirm)}
    baseline=base if policies['global']['selected']=='reference' else query[policies['global']['selected']]
    for arm in ('real','checked','deranged','random','mean'):
        policies[arm]=select(d,candidates(outputs,gates,baseline,arm),baseline,stats,cal,confirm)
    out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'model.npz',**model)
    write_json(out/'policy.json',policies)
    write_json(out/'contract.json',dict(dataset=name,source=str(SOURCE/name),primary='checked',
        model_sha256=sha256(out/'model.npz'),policy_sha256=sha256(out/'policy.json'),
        source_preparation_sha256=sha256(SOURCE/name/'preparation.json'),
        code_sha256={f:sha256('v7/'+f) for f in ['local_pair_adaptation.py','benchmark_local_pair_adaptation.py']},
        train_only_feature_map='PCA8 of full96 history plus physical state and phase; no meta-training',
        prior='TRAIN-fitted 9x12 ridge, penalty10; 12 residual patches interpolated to96',
        cases='Exact original 5+5; no recall change; known residuals relative to the identical frozen Moirai',
        cal_origins=np.unique(d['start'][cal]).tolist(),confirm_origins=np.unique(d['start'][confirm]).tolist(),
        penalties=list(PENALTIES),betas=list(BETAS),seconds=time.perf_counter()-start,
        caveats=['Fast closed-form pilot, not the proposed full episodic meta-training',
                 'Original retriever was TRAIN-fitted and reused old VAL',
                 'Previously examined sampled TEST: exploratory, not a new blind benchmark',
                 'Support check uses old labels known now, not independent chronological refitting of the global prior',
                 'Current-history prefix self-test not implemented in this fast pilot']))
    print('LOCAL FROZEN',name,{k:v['selected'] for k,v in policies.items()},round(time.perf_counter()-start,1),flush=True)


def evaluate(name,out):
    contract=read(out/'contract.json')
    if sha256(out/'model.npz')!=contract['model_sha256'] or sha256(out/'policy.json')!=contract['policy_sha256']:raise ValueError('Changed frozen local model')
    for f,h in contract['code_sha256'].items():
        if sha256('v7/'+f)!=h:raise ValueError('Changed code after policy freeze')
    with np.load(out/'model.npz',allow_pickle=False) as f:model={k:f[k].copy() for k in f.files}
    d,stats,meta=load(name,'test');outputs,gates=forecasts(model,d,stats,meta['period']);policies=read(out/'policy.json')
    base=d['raw_quantiles'][:,4];gp=policies['global']['selected']
    baseline=base if gp=='reference' else base+float(gp)*(outputs['global_raw']-base)
    results={'moirai':base,'no_rag':baseline};alphas={}
    for arm in ('real','checked','deranged','random','mean'):
        c=candidates(outputs,gates,baseline,arm);key=policies[arm]['selected']
        results[arm]=baseline if key=='reference' else c[key]
        # Uniformly fixed diagnostic, never a per-dataset TEST winner.
        raw=outputs['mean_transfer'] if arm=='mean' else outputs[f"{'real' if arm=='checked' else arm}_10.0"]
        results['raw_'+arm]=np.where(gates['10.0'][:,None],raw,baseline) if arm=='checked' else raw
    saved=RUN_ROOT/name/'predictions.npz';source_run=read(RUN_ROOT/name/'run.json')
    if sha256(saved)!=source_run['predictions_sha256']:raise ValueError('Changed incumbent result')
    with np.load(saved,allow_pickle=False) as f:
        for k in ('sid','start','raw_y','selected_starts'):np.testing.assert_array_equal(d[k],f[k])
        results['restored']=f[PRIMARY].copy()
    sd=np.maximum(np.asarray(stats['std'])[d['sid']],1e-6)
    scores={k:metric(d['raw_y'],v,sd) for k,v in results.items()}
    record=dict(dataset=name,queries=len(base),metrics=scores,policies=policies,
        gate_accept_fraction={k:float(v.mean()) for k,v in gates.items()},
        primary='checked',gain_vs_moirai_percent=100*(1-scores['checked']['mse']/scores['moirai']['mse']),
        gain_vs_no_rag_percent=100*(1-scores['checked']['mse']/scores['no_rag']['mse']))
    np.savez_compressed(out/'predictions.npz',**results,sid=d['sid'],start=d['start'])
    write_json(out/'result.json',record)
    print('LOCAL RESULT',name,'gain_M',round(record['gain_vs_moirai_percent'],3),'extra',round(record['gain_vs_no_rag_percent'],3),flush=True)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS));a=p.parse_args()
    root=Path(a.output)
    if root.exists():raise FileExistsError('Choose a new experiment folder')
    root.mkdir(parents=True);begin=time.perf_counter()
    for name in a.datasets:calibrate(name,root/name)
    write_json(root/'frozen.json',{name:sha256(root/name/'contract.json') for name in a.datasets})
    records=[evaluate(name,root/name) for name in a.datasets]
    write_json(root/'summary.json',dict(complete=True,primary='checked',seconds=time.perf_counter()-begin,records=records))


if __name__=='__main__':main()
