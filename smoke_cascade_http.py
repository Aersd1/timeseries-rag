"""Local HTTP timing, always terminate the task-owned server."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,subprocess,sys,time,urllib.request
from pathlib import Path
import numpy as np
from cascade import Cascade

def request(url,data=None):
    payload=None if data is None else json.dumps(data).encode('utf-8')
    with urllib.request.urlopen(urllib.request.Request(url,data=payload,headers={'Content-Type':'application/json'}),timeout=10) as response:
        return json.load(response)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/cascade_certified_http')
    ap.add_argument('--verification',choices=('none','tree','scan'),default='scan');args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);log=(out/'server.log').open('w',encoding='utf-8')
    server=subprocess.Popen([sys.executable,'serve.py','--index','results/validation_index','--cascade-index','results/cascade_index','--port','18779'],
        stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    c=None
    try:
        url='http://127.0.0.1:18779';deadline=time.monotonic()+30
        while True:
            try:request(url+'/info');break
            except OSError:
                if server.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Server did not start')
                time.sleep(.1)
        c=Cascade('results/cascade_index');rng=np.random.default_rng(20260920);groups={};rows=[]
        c.idx.prepare_fft()
        for sid,s in enumerate(c.idx.manifest['series']):
            if s['n']>800:groups.setdefault(s.get('group','unknown'),[]).append(sid)
        for group,sids in groups.items():
            for case in range(5):
                sid=int(rng.choice(sids));x=c.idx.values(sid);start=int(rng.integers(int(len(x)*.65),len(x)-243));q=x[start:start+244].tolist()
                began=time.perf_counter();r=request(url+'/search',dict(query=q,k=10,include_values=True,verification=args.verification));wall=(time.perf_counter()-began)*1000
                assert r['certified']==(args.verification!='none') and r['exact_distances_in_candidates']
                truth=c.idx.fft_exhaustive(q,k=10);cutoff=truth['hits'][-1]['squared_distance']
                recall=sum(h['squared_distance']<=cutoff+2e-5 for h in r['hits'])/10
                if r['certified']:assert recall==1
                rows.append(dict(group=group,case=case,wall_ms=wall,server_ms=r['total_ms'],hits=len(r['hits']),recall=recall))
        original=json.loads(Path('../rag-search/results/full_corpus/query.json').read_text(encoding='utf-8'))
        began=time.perf_counter();r=request(url+'/search',dict(query=original['values'],k=10,include_values=True,verification=args.verification));wall=(time.perf_counter()-began)*1000
        (out/'original_query.json').write_text(json.dumps(dict(http_ms=wall,result=r),ensure_ascii=False,indent=2),encoding='utf-8')
        ms=[r['wall_ms'] for r in rows];summary=dict(queries=len(ms),seed=20260920,p50_ms=float(np.median(ms)),
            p95_ms=float(np.percentile(ms,95)),max_ms=max(ms),over100=sum(v>100 for v in ms),verification=args.verification,
            recall=float(np.mean([r['recall'] for r in rows])),
            timing='Local HTTP round trip, connection per request, client encode/decode included; startup excluded; single client, no concurrency load',rows=rows)
        (out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
    finally:
        if c is not None:c.idx.close()
        server.terminate()
        try:server.wait(timeout=10)
        except subprocess.TimeoutExpired:server.kill();server.wait()
        log.close()

if __name__=='__main__':main()
