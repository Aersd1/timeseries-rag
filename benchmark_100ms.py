"""Ordinary ED retrieval: independent random queries, wall time and completion reported separately."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,time,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from index import SearchIndex

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',default='results/validation_index')
    ap.add_argument('--output',default='results/ordinary_100ms');ap.add_argument('--per-group',type=int,default=5)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    idx=SearchIndex(args.index);t=time.perf_counter();idx.warm_metadata()
    warm_ms=(time.perf_counter()-t)*1000;idx.prepare_fft()
    rng=np.random.default_rng(20260918);groups={};rows=[];example=None
    for sid,s in enumerate(idx.manifest['series']):
        if s['n']>800:groups.setdefault(s.get('group','unknown'),[]).append(sid)
    for group,sids in groups.items():
        for case in range(args.per_group):
            sid=int(rng.choice(sids));x=idx.values(sid);start=int(rng.integers(int(len(x)*.65),len(x)-243))
            q=np.array(x[start:start+244]);target=idx.manifest['series'][sid]
            truth=idx.fft_exhaustive(q,k=10)
            for name,k,budget in [('exact Top-1',1,None),('exact Top-10',10,None),('80ms Top-10',10,80.)]:
                began=time.perf_counter();r=idx.search(q,k=k,time_budget_ms=budget,include_values=True)
                # Includes result construction and JSON serialization, unlike kernel-only latency.
                json.dumps(r);wall=(time.perf_counter()-began)*1000
                cutoff=truth['hits'][k-1]['squared_distance']
                recall=sum(h['squared_distance']<=cutoff+2e-5 for h in r['hits'])/k
                source=any(h.get('source')==target.get('source') and h.get('column')==target.get('column') and
                    h.get('device')==target.get('device') and h['start']==target.get('start',0)+start for h in r['hits'])
                rows.append(dict(group=group,case=case,method=name,wall_ms=wall,recall=recall,source_hit=source,
                    certified=r['certified'],positions_verified=r['positions_verified'],constant=bool(q.std()<=1e-10)))
                if example is None and r['hits']:
                    example=dict(query=q.tolist(),match=r['hits'][0].get('values'),group=group)
            print(group,case,round(rows[-2]['wall_ms'],2),round(rows[-1]['wall_ms'],2),rows[-1]['certified'],flush=True)
    summary=[]
    for method in dict.fromkeys(r['method'] for r in rows):
        rr=[r for r in rows if r['method']==method];ms=[r['wall_ms'] for r in rr]
        summary.append(dict(method=method,queries=len(rr),p50_ms=float(np.median(ms)),p95_ms=float(np.percentile(ms,95)),
            max_ms=max(ms),over_100ms=sum(v>100 for v in ms),recall=float(np.mean([r['recall'] for r in rr])),
            min_recall=min(r['recall'] for r in rr),certified_rate=float(np.mean([r['certified'] for r in rr])),
            source_hit_rate=float(np.mean([r['source_hit'] for r in rr]))))
    report=dict(index=str(Path(args.index).resolve()),windows=idx.manifest['windows'],scope=idx.manifest['scope'],
        query_kind='Unmodified random source slices, including constants; z-normalized Euclidean distance',
        seed=20260918,warm_metadata_ms=warm_ms,summary=summary,rows=rows,
        timing='Resident in-process API plus result JSON serialization; excludes startup and network; no hard real-time guarantee')
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for s in summary:
        vals=sorted(r['wall_ms'] for r in rows if r['method']==s['method'])
        ax[0].plot(np.arange(1,len(vals)+1)/len(vals)*100,vals,label=s['method'])
    ax[0].axhline(100,color='red',ls='--');ax[0].set(xlabel='Queries (%)',ylabel='Wall time (ms)',title='Ordinary retrieval latency');ax[0].legend()
    ax[1].bar(np.arange(3)-.18,[s['recall'] for s in summary],width=.36,label='Distance Recall@k')
    ax[1].bar(np.arange(3)+.18,[s['certified_rate'] for s in summary],width=.36,label='Complete search fraction')
    ax[1].set_xticks(range(3),[s['method'] for s in summary]);ax[1].set_ylim(0,1.1);ax[1].legend()
    fig.suptitle(f'{idx.manifest["windows"]:,} positions; NOT a 10B-point benchmark');fig.tight_layout();fig.savefig(out/'latency.png',dpi=160)
    print(json.dumps(summary,indent=2));idx.close()

if __name__=='__main__':main()
