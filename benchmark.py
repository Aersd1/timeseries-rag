"""Held-out locations, explicit source recovery, and independent exhaustive oracles."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from index import SearchIndex
from model import znorm
import kernels

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',default='results/validation_index')
    ap.add_argument('--output',default='results/benchmark');ap.add_argument('--per-group',type=int,default=1)
    ap.add_argument('--dtw-queries',type=int,default=7)
    ap.add_argument('--seed',type=int,default=20260916)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    idx=SearchIndex(args.index);rng=np.random.default_rng(args.seed);groups={};records=[];examples=[]
    for sid,info in enumerate(idx.manifest['series']):
        if info['n']>=800:groups.setdefault(info.get('group','unknown'),[]).append(sid)
    bases=[];rejected=0
    for group,sids in groups.items():
        for _ in range(args.per_group):
            for attempt in range(300):
                sid=int(rng.choice(sids));x=idx.values(sid)
                start=int(rng.integers(int(len(x)*.65),len(x)-243));q=np.array(x[start:start+244])
                if q.std()>1e-7:
                    bases.append((group,sid,start,q));break
                rejected+=1
            else:raise RuntimeError(f'No nonconstant held-out query in {group}')
    # Separate first-process query from repeated resident latency. No OS cache flush is claimed.
    first=idx.search(bases[0][3],k=10);repeat=[idx.search(bases[0][3],k=10)['total_ms'] for _ in range(3)]
    idx.prepare_fft();case=0;dtw_done=0
    for group,sid,start,original in bases:
        sd=original.std();t=np.arange(244)
        variants=[('original',original),('noise5',original+rng.normal(0,sd*.05,244)),
            ('noise10',original+rng.normal(0,sd*.1,244)),('affine',original*1.7+sd*3),
            ('warp6',np.interp(t+6*np.sin(2*np.pi*t/243),t,original))]
        for kind,q in variants:
            metrics=['ed']+(['dtw'] if kind=='warp6' and dtw_done<args.dtw_queries else [])
            for metric in metrics:
                if metric=='dtw':dtw_done+=1
                truth=idx.exhaustive(q,k=10,metric=metric,radius=8)
                z,_,_,env=idx.model.query(q,metric,8)
                origin_distance=float(kernels.exact(idx.values(sid),np.array([start]),z,metric,8,np.inf,env)[0][0])
                trials=[('native exhaustive',truth,10)]
                if metric=='ed':trials.append(('full FFT',idx.fft_exhaustive(q,k=10),10))
                trials += [('tree exact k1',idx.search(q,k=1,metric=metric),1),
                    ('tree exact k10',idx.search(q,k=10,metric=metric,include_values=True),10),
                    ('tree 900ms k10',idx.search(q,k=10,metric=metric,time_budget_ms=900),10)]
                target=idx.manifest['series'][sid];target_start=target.get('start',0)+start
                for method,result,k in trials:
                    cutoff=truth['hits'][k-1]['squared_distance']
                    recall=sum(h['squared_distance']<=cutoff+2e-5 for h in result['hits'])/k
                    source_hit=any(h.get('source')==target.get('source') and h.get('column')==target.get('column') and
                        h.get('device')==target.get('device') and abs(h['start']-target_start)<=(8 if kind=='warp6' else 0)
                        for h in result['hits'])
                    if method.startswith('tree exact') and recall<.999999:
                        raise AssertionError(f'Certified search disagrees with exhaustive oracle: {group}, {kind}, {metric}, {recall}')
                    row=dict(case=case,group=group,kind=kind,metric=metric,method=method,k=k,
                        ms=result['total_ms'],recall=recall,source_hit=source_hit,
                        origin_distance=origin_distance,origin_is_topk_distance=origin_distance<=cutoff+2e-5,
                        best_distance=result['hits'][0]['squared_distance'] if result['hits'] else None,
                        certified=result.get('certified',True),budget_exhausted=result.get('budget_exhausted',False),
                        positions_verified=result.get('positions_verified',idx.manifest['windows']),
                        nodes_visited=result.get('nodes_visited'),shards_opened=result.get('shards_opened'))
                    records.append(row)
                    if method=='tree exact k10' and (metric=='dtw' or len(examples)<2):
                        examples.append(dict(group=group,kind=kind,metric=metric,query=q.tolist(),
                            original=original.tolist(),true_start=target_start,hits=result['hits'][:3]))
                print(f'{case+1}: {group}/{kind}/{metric}; exact {trials[-2][1]["total_ms"]:.1f}ms; verified',flush=True)
                case+=1
                (out/'progress.json').write_text(json.dumps(dict(completed_cases=case,records=records),ensure_ascii=False),encoding='utf-8')
    summary=[]
    for metric in ('ed','dtw'):
        for method in dict.fromkeys(r['method'] for r in records if r['metric']==metric):
            rr=[r for r in records if r['metric']==metric and r['method']==method];ms=[r['ms'] for r in rr]
            summary.append(dict(metric=metric,method=method,queries=len(rr),p50_ms=float(np.median(ms)),
                p95_ms=float(np.percentile(ms,95)),max_ms=max(ms),recall=float(np.mean([r['recall'] for r in rr])),
                source_recall=float(np.mean([r['source_hit'] for r in rr])),certified_rate=float(np.mean([r['certified'] for r in rr])),
                min_recall=min(r['recall'] for r in rr),median_positions=float(np.median([r['positions_verified'] for r in rr]))))
    report=dict(index=str(Path(args.index).resolve()),windows=idx.manifest['windows'],stored_points=idx.manifest['stored_points'],
        records_count=idx.manifest['records'],shards=len(idx.manifest['shards']),scope=idx.manifest['scope'],
        seed=args.seed,training_region='first 55% of each run',evaluation_region='starts after 65%; disjoint from training windows',
        rejected_constant_candidates=rejected,first_process_query_ms=first['total_ms'],same_query_resident_ms=repeat,
        summary=summary,records=records,examples=examples)
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    for ri,metric in enumerate(('ed','dtw')):
        ss=[s for s in summary if s['metric']==metric];labels=[s['method'].replace(' ','\n',1) for s in ss]
        axes[ri,0].bar(labels,[s['p50_ms'] for s in ss]);axes[ri,0].scatter(labels,[s['p95_ms'] for s in ss],color='black',label='p95')
        axes[ri,0].axhline(1000,color='red',linestyle='--',label='1s');axes[ri,0].set_yscale('log');axes[ri,0].legend()
        axes[ri,0].set_title(f'{metric.upper()} latency (ms, log scale)')
        axes[ri,1].bar(labels,[s['recall']*100 for s in ss],label='distance Recall@k')
        axes[ri,1].scatter(labels,[s['source_recall']*100 for s in ss],color='darkorange',label='true source hit@k')
        axes[ri,1].set_ylim(0,105);axes[ri,1].legend();axes[ri,1].set_title(f'{metric.upper()} quality (%)')
        for ax in axes[ri]:ax.tick_params(axis='x',labelsize=8)
    fig.suptitle(f'Learned hierarchical index: {idx.manifest["windows"]:,} positions (NOT 10B-scale validation)')
    fig.tight_layout();fig.savefig(out/'latency_quality.png',dpi=160);plt.close(fig)
    examples=[e for e in examples if e['metric']=='dtw'][:3] or examples[:3]
    fig,axes=plt.subplots(len(examples),1,figsize=(12,3.5*len(examples)),squeeze=False)
    for ax,e in zip(axes[:,0],examples):
        ax.plot(znorm(e['original']),label='Original source',linewidth=2)
        ax.plot(znorm(e['query']),label='Warped query',alpha=.8)
        if e['hits']:ax.plot(znorm(e['hits'][0]['values']),label=f"Retrieved @ {e['hits'][0]['start']}",linestyle='--')
        ax.set_title(f"{e['group']} / {e['metric']}; original start {e['true_start']}");ax.legend()
    fig.tight_layout();fig.savefig(out/'localization.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True);idx.close()

if __name__=='__main__':main()
