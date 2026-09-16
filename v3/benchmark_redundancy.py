import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import kernels
from diverse import DiverseIndex
from benchmark_diverse import overlaps

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',default='../results/cascade_index')
    ap.add_argument('--output',default='results/redundancy');ap.add_argument('--per-group',type=int,default=2)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    engine=DiverseIndex(args.index);rng=np.random.default_rng(20260921);groups={};rows=[]
    for sid,s in enumerate(engine.idx.manifest['series']):
        if s['n']>800:groups.setdefault(s.get('group','unknown'),[]).append(sid)
    try:
        for group,sids in groups.items():
            for case in range(args.per_group):
                sid=int(rng.choice(sids));x=engine.idx.values(sid);p=int(rng.integers(int(len(x)*.65),len(x)-243));q=np.array(x[p:p+244])
                # Audit-only full distances. Never used by the retrieval engine;
                # compute once outside timing to validate all configurations.
                z,_,_,_=engine.idx.model.query(q)
                truth=[kernels.exact(engine.idx.values(i),np.arange(s['n']-243,dtype=np.int64),z)[0]
                    for i,s in enumerate(engine.idx.manifest['series'])]
                configs=[('old adaptive128',None),('bounded128',128),('bounded512',512),('bounded2048',2048),('bounded4384',4384)]
                # Rotate execution order, deterministic, to reduce fixed order bias.
                shift=(len(rows)//5)%5;configs=configs[shift:]+configs[:shift]
                for name,initial in configs:
                    began=time.perf_counter()
                    r=engine.search_prefix(q,include_values=True) if initial is None else engine.search_bounded(q,initial_candidates=initial,include_values=True)
                    json.dumps(r);wall=(time.perf_counter()-began)*1000
                    available=[d.copy() for d in truth]
                    assert r['certified'] and len(r['hits'])==10 and not overlaps(engine,r['hits'])
                    for hit in r['hits']:
                        best=min(float(d.min()) for d in available)
                        assert abs(best-hit['squared_distance'])<=2e-5,(name,group,best,hit['squared_distance'])
                        for i,d in enumerate(available):
                            if engine.keys[i]==engine.keys[hit['sid']]:
                                positions=np.arange(len(d))+engine.offsets[i]
                                d[abs(positions-hit['start'])<244]=np.inf
                    rows.append(dict(group=group,case=case,method=name,ms=wall,rounds=len(r['prefix_rounds']),
                        final_candidates=r['prefix_rounds'][-1]['requested'],verified=r['positions_verified'],recall=1.,
                        retained=r.get('max_retained_candidates'),tracked_numeric_bytes=r.get('tracked_numeric_peak_bytes'),
                        candidate_bytes=r.get('candidate_payload_bound_bytes')))
                print(group,case,[(r['method'],round(r['ms'],1),r['rounds']) for r in rows[-5:]],flush=True)
                (out/'progress.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        summary=[]
        for name in ('old adaptive128','bounded128','bounded512','bounded2048','bounded4384'):
            rr=[r for r in rows if r['method']==name];ms=[r['ms'] for r in rr]
            summary.append(dict(method=name,queries=len(rr),p50_ms=float(np.median(ms)),p95_ms=float(np.percentile(ms,95)),
                max_ms=max(ms),over100=sum(v>100 for v in ms),recall=1.,mean_rounds=float(np.mean([r['rounds'] for r in rr])),
                max_rounds=max(r['rounds'] for r in rr),refill_queries=sum(r['rounds']>1 for r in rr),
                max_candidates=max(r['final_candidates'] for r in rr),
                max_tracked_numeric_bytes=max((r['tracked_numeric_bytes'] or 0) for r in rr)))
        (out/'metrics.json').write_text(json.dumps(dict(windows=engine.base.meta['windows'],seed=20260921,
            summary=summary,rows=rows,scope='Development corpus; resident calls plus JSON serialization; audit-only full distances excluded from query timing/memory'),indent=2),encoding='utf-8')
        fig,ax=plt.subplots(1,3,figsize=(15,4.5));labels=[s['method'].replace(' ','\n').replace('bounded','bounded\n') for s in summary]
        ax[0].bar(labels,[s['p50_ms'] for s in summary]);ax[0].scatter(labels,[s['p95_ms'] for s in summary],color='black',label='p95');ax[0].axhline(100,color='red',ls='--');ax[0].set_title('Latency (ms)');ax[0].legend()
        ax[1].bar(labels,[s['mean_rounds'] for s in summary]);ax[1].set_title('Average full-prefix passes')
        ax[2].bar(labels[1:],[s['max_tracked_numeric_bytes']/1024 for s in summary[1:]]);ax[2].set_title('Tracked numeric buffers (KiB)\nNot process RSS / not index memory')
        fig.suptitle('Exact non-overlapping top-10; 14 queries if per-group=2');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160)
        print(json.dumps(summary,indent=2))
    finally:engine.close()

if __name__=='__main__':main()
