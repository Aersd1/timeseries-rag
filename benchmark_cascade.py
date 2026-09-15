import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,time,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cascade import Cascade

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',default='results/cascade_index');ap.add_argument('--output',default='results/cascade_benchmark')
    ap.add_argument('--per-group',type=int,default=10);ap.add_argument('--certify',action='store_true')
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    obj=Cascade(args.index);idx=obj.idx;idx.warm_metadata();idx.prepare_fft();rng=np.random.default_rng(20260919)
    groups={};rows=[];example=None
    for sid,s in enumerate(idx.manifest['series']):
        if s['n']>800:groups.setdefault(s.get('group','unknown'),[]).append(sid)
    for group,sids in groups.items():
        for case in range(args.per_group):
            sid=int(rng.choice(sids));x=idx.values(sid);start=int(rng.integers(int(len(x)*.65),len(x)-243));q=np.array(x[start:start+244])
            target=idx.manifest['series'][sid];truth=idx.fft_exhaustive(q,k=10);cutoff=truth['hits'][-1]['squared_distance']
            trials=[('exact tree',lambda:idx.search(q,k=10,include_values=True))]
            if args.certify:
                for ver in ('none','tree','scan'):
                    trials.append((f'cascade {ver}',lambda ver=ver:obj.search(q,include_values=True,verification=ver)))
            else:
                for l,m in [(4,4),(12,16),(24,32)]:trials.append((f'cascade {l}/{m}',lambda l=l,m=m:obj.search(q,long_candidates=l,mid_candidates=m,include_values=True,verification='none')))
            for name,fn in trials:
                began=time.perf_counter();r=fn();json.dumps(r);wall=(time.perf_counter()-began)*1000
                recall=sum(h['squared_distance']<=cutoff+2e-5 for h in r['hits'])/10
                source=any(h.get('source')==target.get('source') and h.get('column')==target.get('column') and
                    h.get('device')==target.get('device') and h['start']==target.get('start',0)+start for h in r['hits'])
                coarse=any(int(obj.blocks[b][0])==sid and obj.blocks[b][1]<=start<obj.blocks[b][2] for b in r.get('selected_long_blocks',[]))
                fine=any(b['sid']==sid and b['start']<=start<b['end'] for b in r.get('selected_middle_blocks',[]))
                if name=='exact tree':
                    coarse=fine=True
                if r.get('certified') and recall<1:raise AssertionError('Certified search disagrees with FFT')
                rows.append(dict(group=group,case=case,method=name,ms=wall,recall=recall,source_hit=source,
                    coarse_hit=bool(coarse),middle_hit=bool(fine),positions=r['positions_verified']+r.get('verification_positions',0),
                    certified=r.get('certified',False),verification_ms=r.get('verification_ms'),
                    symbol_positions=r.get('symbol_positions_checked'),coarse_ms=r.get('coarse_ms'),middle_ms=r.get('middle_ms')))
                if name in ('cascade 12/16','cascade scan') and example is None and source:example=dict(query=q.tolist(),hit=r['hits'][0],group=group)
            print(group,case,[(r['method'],round(r['ms'],1),r['recall']) for r in rows[-4:]],flush=True)
    summary=[]
    for method in dict.fromkeys(r['method'] for r in rows):
        rr=[r for r in rows if r['method']==method];ms=[r['ms'] for r in rr]
        summary.append(dict(method=method,queries=len(rr),p50_ms=float(np.median(ms)),p95_ms=float(np.percentile(ms,95)),max_ms=max(ms),
            over100=sum(v>100 for v in ms),recall=float(np.mean([r['recall'] for r in rr])),min_recall=min(r['recall'] for r in rr),
            certified_rate=float(np.mean([r['certified'] for r in rr])),
            source_hit=float(np.mean([r['source_hit'] for r in rr])),coarse_hit=float(np.mean([r['coarse_hit'] for r in rr])),
            middle_hit=float(np.mean([r['middle_hit'] for r in rr])),median_verified=float(np.median([r['positions'] for r in rr]))))
    report=dict(seed=20260919,windows=obj.meta['windows'],scope=idx.manifest['scope'],summary=summary,rows=rows,
        timing='Resident API plus result serialization, no network or startup; unmodified queries; same cases for every configuration',example=example)
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(1,2,figsize=(13,4.5))
    for s in summary:
        vals=sorted(r['ms'] for r in rows if r['method']==s['method']);ax[0].plot(np.linspace(0,100,len(vals)),vals,label=s['method'])
    ax[0].axhline(100,color='red',ls='--');ax[0].set_yscale('log');ax[0].set(xlabel='Queries (%)',ylabel='Latency ms (log)',title='Same-query comparison');ax[0].legend()
    xx=np.arange(len(summary))
    for delta,key,label in [(-.25,'coarse_hit','True long block'),(0,'middle_hit','True middle block'),(.25,'recall','Distance Recall@10')]:
        ax[1].bar(xx+delta,[s[key] for s in summary],width=.25,label=label)
    ax[1].set_xticks(xx,[s['method'].replace(' ','\n') for s in summary]);ax[1].set_ylim(0,1.15);ax[1].legend(loc='lower left')
    fig.suptitle(f'{obj.meta["windows"]:,} positions; {len(rows)//4} original queries; NOT 10B scale');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160)
    if example:
        fig,ax=plt.subplots(figsize=(10,3));ax.plot(example['query'],label='Query');ax.plot(example['hit']['values'],ls='--',label='Retrieved');ax.legend();ax.set_title('244-point localization: '+example['group']);fig.tight_layout();fig.savefig(out/'localization.png',dpi=160)
    print(json.dumps(summary,indent=2));idx.close()

if __name__=='__main__':main()
