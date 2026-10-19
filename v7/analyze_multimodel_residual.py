"""Predefined H1 analysis: paired seeds, synchronized time blocks,12-test family."""
import argparse,itertools
from pathlib import Path
import numpy as np
from .common import write_json,sha256
from .benchmark_memory_access import read
from .benchmark_multimodel_residual import arrays,model_path
from .multimodel_residual import MODELS,H

SCORE_ARMS=('base','query_only','real','matched_random','deranged','memory_mean')
COMPARISONS=('query_only','matched_random','deranged')


def holm(pvalues):
    order=np.argsort(pvalues);adjusted=np.empty(len(pvalues));running=0.
    for rank,index in enumerate(order):
        running=max(running,(len(pvalues)-rank)*pvalues[index]);adjusted[index]=min(1.,running)
    return adjusted


def resampled_risks(sums,counts,seed_ids,week_ids):
    selected=sums[seed_ids].mean(1)  # replicate,week,arm
    picked=selected[np.arange(len(selected))[:,None],week_ids]
    return picked.sum(1)/counts[week_ids].sum(1)[:,None]


def analyze(root,replicates=10000):
    summary=read(root/'summary.json');p=summary['protocol'];names=p['datasets'];seeds=p['seeds']
    if not summary['complete']:raise ValueError('Incomplete experiment')
    cells=[];macro=[];test_family=[];macro_draws={};memory_family=[]
    for kind in p['models']:
        rng=np.random.default_rng(20261019)
        seed_ids=rng.integers(len(seeds),size=(replicates,len(seeds)))
        ds_points=[];ds_draws=[]
        for name in names:
            folder=root/name;packet=read(folder/'packet.json');all_scores=[];raw=[];mae=[];origins=None
            for seed in seeds:
                path=model_path(folder,kind,seed);record=read(path/f'test_{seed}.json')
                if sha256(path/f'test_rows_{seed}.npz')!=record['rows_sha256']:raise ValueError('Changed scored rows')
                d=arrays(path/f'test_rows_{seed}.npz');starts=np.unique(d['start']);sids=np.unique(d['sid'])
                if origins is not None:np.testing.assert_array_equal(origins,starts)
                origins=starts
                for sid in sids:np.testing.assert_array_equal(d['start'][d['sid']==sid],starts)
                values=[]
                for arm in SCORE_ARMS:
                    if arm=='deranged':v=np.mean([d[f'loss_deranged_{j}'] for j in (1,2,3,4)],0)
                    else:v=d['loss_'+arm]
                    values.append(np.stack([v[d['start']==t].mean(0) for t in origins]))
                values=np.stack(values,1)  # time,arm,metric
                all_scores.append(values[:,:,2]);raw.append(values[:,:,0].mean(0));mae.append(values[:,:,1].mean(0))
            all_scores=np.stack(all_scores);point=all_scores.mean((0,1));ds_points.append(point)
            block_samples=max(1,int(round(7*packet['daily_period'])))
            block_samples*=int(np.ceil((512+96)/block_samples))
            week=(origins+512)//block_samples
            groups=np.unique(week);counts=np.array([(week==w).sum() for w in groups])
            sums=np.stack([all_scores[:,week==w].sum(1) for w in groups],1)
            week_ids=rng.integers(len(groups),size=(replicates,len(groups)))
            draws=resampled_risks(sums,counts,seed_ids,week_ids);ds_draws.append(draws)
            comparisons={}
            for arm in ('base',)+COMPARISONS:
                j=SCORE_ARMS.index(arm);g=100*(1-draws[:,2]/np.maximum(draws[:,j],1e-30))
                comparisons[arm]=dict(gain_percent=float(100*(1-point[2]/max(point[j],1e-30))),
                    interval95=np.quantile(g,[.025,.975]).tolist(),
                    seed_gains_percent=(100*(1-all_scores[:,:,2].mean(1)/np.maximum(all_scores[:,:,j].mean(1),1e-30))).tolist())
            cells.append(dict(dataset=name,model=kind,queries=record['queries'],variables=record['variables'],
                origins=len(origins),time_blocks=len(groups),block_days=block_samples/packet['daily_period'],
                insufficient_time_blocks=len(groups)<8,risk_nmse=dict(zip(SCORE_ARMS,point.tolist())),
                raw_mse=dict(zip(SCORE_ARMS,np.mean(raw,0).tolist())),mae=dict(zip(SCORE_ARMS,np.mean(mae,0).tolist())),
                real_vs=comparisons))
        points=np.mean(ds_points,0);draws=np.mean(ds_draws,0);macro_draws[kind]=draws
        sensitivity_weights=np.array([.25/4 if n.startswith('ETT') else .25 for n in names])
        sensitivity_weights/=sensitivity_weights.sum()
        domain_point=np.average(ds_points,axis=0,weights=sensitivity_weights)
        m=dict(model=kind,risk_nmse=dict(zip(SCORE_ARMS,points.tolist())),
            ett_as_one_domain_risk=dict(zip(SCORE_ARMS,domain_point.tolist())),tests={})
        for arm in COMPARISONS:
            j=SCORE_ARMS.index(arm);observed=points[j]-points[2]
            bootstrap_delta=draws[:,j]-draws[:,2]
            # One-sided centered paired-bootstrap null. Assumes the time-block
            # resampling model; this is exploratory, not an exact randomization test.
            pvalue=(1+np.sum(bootstrap_delta-observed>=observed))/(replicates+1)
            gains=100*(1-draws[:,2]/np.maximum(draws[:,j],1e-30))
            test=dict(gain_percent=float(100*(1-points[2]/max(points[j],1e-30))),
                bootstrap_p_one_sided=float(pvalue),interval95=np.quantile(gains,[.025,.975]).tolist(),
                familywise_interval95=np.quantile(gains,[.05/(2*12),1-.05/(2*12)]).tolist(),
                domain_sensitivity_gain_percent=float(100*(1-domain_point[2]/max(domain_point[j],1e-30))))
            m['tests'][arm]=test;test_family.append(test)
        observed=points[5]-points[2];delta=draws[:,5]-draws[:,2]
        g=100*(1-draws[:,2]/np.maximum(draws[:,5],1e-30))
        m['memory_matching_necessity_test']=dict(gain_percent=float(100*(1-points[2]/max(points[5],1e-30))),
            bootstrap_p_one_sided=float((1+np.sum(delta-observed>=observed))/(replicates+1)),
            interval95=np.quantile(g,[.025,.975]).tolist(),familywise_interval95=np.quantile(g,[.05/(2*4),1-.05/(2*4)]).tolist())
        memory_family.append(m['memory_matching_necessity_test'])
        macro.append(m)
    adjusted=holm([t['bootstrap_p_one_sided'] for t in test_family])
    for test,pvalue in zip(test_family,adjusted):
        test['holm_p']=float(pvalue)
        test['meets_practical_and_statistical_threshold']=bool(test['gain_percent']>=1 and pvalue<.05 and test['familywise_interval95'][0]>0)
    for m in macro:
        m['supports_rag_extra_in_this_probe']=all(t['meets_practical_and_statistical_threshold'] for t in m['tests'].values())
        m['real_gain_vs_base_percent']=100*(1-m['risk_nmse']['real']/max(m['risk_nmse']['base'],1e-30))
        m['adequate_blocks_in_all_datasets']=all(not c['insufficient_time_blocks'] for c in cells if c['model']==m['model'])
    for test,pvalue in zip(memory_family,holm([t['bootstrap_p_one_sided'] for t in memory_family])):
        test['holm_p_separate_secondary_family']=float(pvalue)
        test['supports_matching_beyond_global_memory']=bool(test['gain_percent']>=1 and pvalue<.05 and test['familywise_interval95'][0]>0)
    interactions=[]
    for first,second in itertools.combinations(p['models'],2):
        a=next(m for m in macro if m['model']==first);b=next(m for m in macro if m['model']==second)
        point=(a['risk_nmse']['query_only']-a['risk_nmse']['real'])-(b['risk_nmse']['query_only']-b['risk_nmse']['real'])
        draw=(macro_draws[first][:,1]-macro_draws[first][:,2])-(macro_draws[second][:,1]-macro_draws[second][:,2])
        centered=draw-point
        pvalue=(1+np.sum(abs(centered)>=abs(point)))/(replicates+1)
        interactions.append(dict(first=first,second=second,delta_nmse_difference=float(point),
            interval95=np.quantile(draw,[.025,.975]).tolist(),p_two_sided=float(pvalue)))
    for item,adjusted_p in zip(interactions,holm([i['p_two_sided'] for i in interactions])):item['holm_p_separate_secondary_family']=float(adjusted_p)
    report=dict(primary='H1 fixed-model residual predictability; all512-to96',replicates=replicates,
        macro=macro,cells=cells,paired_model_interactions=interactions,
        independent_unit='Synchronized blocks across all channels:7 days times ceil608/(7*daily_period); paired seed resampling; cells below8 blocks flagged',
        multiple_tests='12 one-sided centered bootstrap tests with Holm p; conservative Bonferroni simultaneous percentile intervals',
        caveats=['Already explored datasets: exploratory protocol-based replication, not fresh blind confirmation',
            'Intervals depend on time-block assumptions; only3 training/probe seeds',
            'Only three architecture families; no cross-TSFM generalization claim',
            'Failure of this128-feature probe does not prove no useful information exists',
            'H2-H5 full interventions and four-level expected-utility gates are not executed by the H1 runner'])
    write_json(root/'analysis.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    r=analyze(Path(p.parse_args().root));print([(m['model'],m['supports_rag_extra_in_this_probe']) for m in r['macro']])
