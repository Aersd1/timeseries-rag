"""Resident local/worker service. One process owns an explicit subset of shards."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,hashlib,json
from http.server import HTTPServer,BaseHTTPRequestHandler
from pathlib import Path
import numpy as np
from index import SearchIndex

def model_id(model):
    return hashlib.sha256(model.freq.tobytes()+model.bins.tobytes()+str(model.length).encode()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',required=True);ap.add_argument('--port',type=int,default=8770)
    ap.add_argument('--shard-start',type=int,default=0);ap.add_argument('--shard-end',type=int)
    ap.add_argument('--cache-shards',type=int,default=256);ap.add_argument('--cold',action='store_true')
    ap.add_argument('--cascade-index',help='Use approximate long/middle/point routing for ED queries')
    args=ap.parse_args();manifest=json.loads((Path(args.index)/'manifest.json').read_text(encoding='utf-8'))
    end=args.shard_end if args.shard_end is not None else len(manifest['shards'])
    idx=SearchIndex(args.index,cache_shards=args.cache_shards,owned_shards=range(args.shard_start,end))
    cascade=None
    if args.cascade_index:
        if args.shard_start!=0 or end!=len(manifest['shards']):raise ValueError('Cascade service currently requires full index ownership')
        from cascade import Cascade
        cascade=Cascade(args.cascade_index)
        if Path(cascade.meta['base']).resolve()!=Path(args.index).resolve():raise ValueError('Cascade and raw index mismatch')
    if not args.cold and cascade is None:idx.warm_metadata()
    lo=idx.root_lo[idx.owned_shards].min(axis=0).tolist();hi=idx.root_hi[idx.owned_shards].max(axis=0).tolist()
    info=dict(index_id=hashlib.sha256((Path(args.index)/'manifest.json').read_bytes()).hexdigest(),
        model_id=model_id(idx.model),owned_shards=idx.owned_shards,total_shards=len(manifest['shards']),
        windows=idx.search_windows,lower_codes=lo,upper_codes=hi,scope=manifest['scope'])
    class Handler(BaseHTTPRequestHandler):
        def reply(self,value,status=200):
            body=json.dumps(value,ensure_ascii=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):self.reply(info)
        def do_POST(self):
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<100000:raise ValueError('Invalid request length')
                request=json.loads(self.rfile.read(n))
                metric=request.get('metric','ed')
                options=dict(k=int(request.get('k',10)),radius=int(request.get('radius',8)),
                    time_budget_ms=request.get('budget_ms',80. if metric=='ed' else None),include_values=bool(request.get('include_values',False)))
                if cascade is not None and metric=='ed':
                    result=cascade.search(request['query'],k=options['k'],include_values=options['include_values'],
                        long_candidates=int(request.get('long_candidates',12)),mid_candidates=int(request.get('mid_candidates',16)),
                        verification=request.get('verification','scan'))
                    result['execution_mode']='cascade + selected verification; budget_ms is not enforced; certification requires completion'
                else:
                    result=idx.robust_search(request['query'],**options) if metric=='robust' else idx.search(request['query'],metric=metric,**options)
                result['index_id']=info['index_id'];self.reply(result)
            except (ValueError,KeyError,TypeError) as exc:self.reply({'error':str(exc)},400)
    print(f'Ready: 127.0.0.1:{args.port}; {len(idx.owned_shards)} shards; {idx.search_windows:,} positions',flush=True)
    try:HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
    finally:
        idx.close()
        if cascade is not None:cascade.idx.close()

if __name__=='__main__':main()
