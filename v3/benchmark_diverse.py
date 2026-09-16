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

def overlaps(engine,hits,separation=244):
    return sum(engine.keys[a['sid']]==engine.keys[b['sid']] and abs(a['start']-b['start'])<separation
        for i,a in enumerate(hits) for b in hits[i+1:])

def audit(engine,q,result):
    z,_,_,_=engine.idx.model.query(q);ds=[]
    for sid,info in enumerate(engine.idx.manifest['series']):
        positions=np.arange(info['n']-len(q)+1,dtype=np.int64)
        ds.append(kernels.exact(engine.idx.values(sid),positions,z)[0])
    successful=0
    for hit in result['hits']:
        best=min(float(d.min()) for d in ds)
        if abs(best-hit['squared_distance'])<=2e-5:successful+=1
        else:raise AssertionError(f'Global remaining-neighbor miss: {best}, {hit["squared_distance"]}')
        for sid,d in enumerate(ds):
            if engine.keys[sid]==engine.keys[hit['sid']]:
                positions=np.arange(len(d))+engine.offsets[sid]
                d[abs(positions-hit['start'])<result['min_separation']]=np.inf
    assert not overlaps(engine,result['hits'],result['min_separation'])
    if len(result['hits'])<result['requested_k']:assert all(not np.isfinite(d).any() for d in ds)
    return successful/max(1,len(result['hits']))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',default='../results/cascade_index')
    ap.add_argument('--output',default='results/benchmark');ap.add_argument('--per-group',type=int,default=2)
    ap.add_argument('--strategy',choices=('hierarchy','prefix'),default='hierarchy')
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    engine=DiverseIndex(args.index);rng=np.random.default_rng(20260921);groups={};rows=[]
    search=engine.search_prefix if args.strategy=='prefix' else engine.search
    for sid,s in enumerate(engine.idx.manifest['series']):
        if s['n']>800:groups.setdefault(s.get('group','unknown'),[]).append(sid)
    try:
        for group,sids in groups.items():
            for case in range(args.per_group):
                sid=int(rng.choice(sids));x=engine.idx.values(sid);p=int(rng.integers(int(len(x)*.65),len(x)-243));q=np.array(x[p:p+244])
                began=time.perf_counter();old=engine.base.search(q,k=10,include_values=True);json.dumps(old);oldms=(time.perf_counter()-began)*1000
                began=time.perf_counter();new=search(q,k=10,include_values=True);json.dumps(new);newms=(time.perf_counter()-began)*1000
                recall=audit(engine,q,new)
                simple=[]
                for h in old['hits']:
                    if not overlaps(engine,simple+[h]):simple.append(h)
                row=dict(group=group,case=case,old_ms=oldms,new_ms=newms,old_overlap_pairs=overlaps(engine,old['hits']),
                    old_postfilter_count=len(simple),new_count=len(new['hits']),greedy_distance_recall=recall,
                    new_overlap_pairs=overlaps(engine,new['hits']),positions_verified=new['positions_verified'],
                    symbol_positions=new['symbol_positions'],middle_expanded=new['middle_expanded'],long_expanded=new['long_expanded'])
                rows.append(row);print(row,flush=True)
                (out/'progress.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        query=json.loads(Path('../results/query.json').read_text(encoding='utf-8'))
        old=engine.base.search(query['values'],include_values=True);new=search(query['values'],include_values=True)
        audit(engine,query['values'],new)
        (out/'gpu_comparison.json').write_text(json.dumps(dict(query=query,ordinary=old,non_overlapping=new),ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'gpu_result.json').write_text(json.dumps(new,ensure_ascii=False,indent=2),encoding='utf-8')
        summary=dict(queries=len(rows),strategy=args.strategy,seed=20260921,windows=engine.base.meta['windows'],startup_ms=engine.startup_ms,
            old_p50_ms=float(np.median([r['old_ms'] for r in rows])),new_p50_ms=float(np.median([r['new_ms'] for r in rows])),
            new_p95_ms=float(np.percentile([r['new_ms'] for r in rows],95)),new_max_ms=max(r['new_ms'] for r in rows),
            new_over100=sum(r['new_ms']>100 for r in rows),greedy_recall=float(np.mean([r['greedy_distance_recall'] for r in rows])),
            overlap_pairs=sum(r['new_overlap_pairs'] for r in rows),rows=rows)
        (out/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        fig,ax=plt.subplots(1,2,figsize=(12,4.5))
        for method in ('old','new'):
            ms=sorted(r[method+'_ms'] for r in rows);ax[0].plot(np.linspace(0,100,len(ms)),ms,label=method)
        ax[0].axhline(100,color='red',ls='--');ax[0].set(xlabel='Queries (%)',ylabel='ms',title='Ordinary vs non-overlapping (different targets)');ax[0].legend()
        ax[1].plot([r['old_postfilter_count'] for r in rows],label='Postfilter ordinary top-10')
        ax[1].plot([r['new_count'] for r in rows],label='Exact greedy non-overlapping');ax[1].set(xlabel='Query',ylabel='Results',ylim=(0,11));ax[1].legend()
        fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
    finally:engine.close()

if __name__=='__main__':main()
