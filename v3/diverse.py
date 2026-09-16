"""Exact greedy, non-overlapping time-series retrieval through a best-first hierarchy."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,bisect,heapq,json,time
from pathlib import Path
import numpy as np
import kernels
from cascade import Cascade

class Exclusions:
    """Merged half-open intervals of forbidden global starts, keyed by logical series."""
    def __init__(self):self.ranges={}
    def contains(self,key,p):
        ranges=self.ranges.get(key,[]);i=bisect.bisect_right(ranges,(int(p),float('inf')))-1
        return i>=0 and p<ranges[i][1]
    def add(self,key,p,separation):
        ranges=list(self.ranges.get(key,[]));bisect.insort(ranges,(int(p)-separation+1,int(p)+separation))
        merged=[]
        for a,b in ranges:
            if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
            else:merged.append((a,b))
        self.ranges[key]=merged

class DiverseIndex:
    def __init__(self,path,batch_size=128):
        if batch_size<1:raise ValueError('Positive batch size required')
        began=time.perf_counter();self.base=Cascade(path);self.idx=self.base.idx;self.batch_size=batch_size
        self.keys=[(os.path.normcase(os.path.normpath(str(s.get('source')))),str(s.get('column')),str(s.get('device'))) for s in self.idx.manifest['series']]
        self.offsets=[int(s.get('start',0)) for s in self.idx.manifest['series']]
        self.middle=[];self.long_children=[];lo=[];hi=[];long_lo=[];long_hi=[]
        for bid,(sid,start,end,offset) in enumerate(self.base.blocks):
            sid,start,end,offset=map(int,(sid,start,end,offset));children=[]
            for a in range(start,end,self.base.meta['mid_size']):
                b=min(end,a+self.base.meta['mid_size']);off=offset+a-start;codes=self.base.codes[off:off+b-a]
                children.append(len(self.middle));self.middle.append((sid,a,b,off,bid))
                lo.append(codes.min(axis=0));hi.append(codes.max(axis=0))
            self.long_children.append(children)
            long_lo.append(np.min([lo[i] for i in children],axis=0));long_hi.append(np.max([hi[i] for i in children],axis=0))
        self.lo=np.asarray(lo,dtype=np.uint8);self.hi=np.asarray(hi,dtype=np.uint8)
        self.long_lo=np.asarray(long_lo,dtype=np.uint8);self.long_hi=np.asarray(long_hi,dtype=np.uint8)
        self.startup_ms=(time.perf_counter()-began)*1000

    def search_prefix(self,query,k=10,min_separation=None,max_distance=None,include_values=False,initial_candidates=128):
        """Certified global prefix, expanded until greedy exclusion yields k results.

        No unseen result can beat a selected result inside a complete nearest
        prefix. This is adaptive refill, not truncating ordinary top-10 once.
        """
        began=time.perf_counter();m=self.idx.model.length;separation=m if min_separation is None else int(min_separation)
        if k<1 or initial_candidates<1 or separation<m:raise ValueError('Invalid result count or separation')
        if max_distance is not None and (not np.isfinite(max_distance) or max_distance<0):raise ValueError('Invalid distance limit')
        limit=np.inf if max_distance is None else max_distance**2
        total=self.base.meta['windows'];count=min(total,max(k,initial_candidates));rounds=[]
        while True:
            found=self.base.search(query,k=count,verification='scan',include_values=False)
            if not found['certified']:raise RuntimeError('Cannot certify an incomplete candidate prefix')
            blocked=Exclusions();chosen=[]
            for h in found['hits']:
                if h['squared_distance']>limit:break
                key=self.keys[h['sid']]
                if blocked.contains(key,h['start']):continue
                chosen.append(h);blocked.add(key,h['start'],separation)
                if len(chosen)>=k:break
            rounds.append(dict(requested=count,returned=len(found['hits']),selected=len(chosen),ms=found['total_ms'],
                positions=found['positions_verified']+found.get('verification_positions',0),
                symbols=found['symbol_positions_checked']+found.get('verification_symbol_positions',0)))
            threshold_exhausted=bool(found['hits']) and found['hits'][-1]['squared_distance']>limit
            if len(chosen)>=k or len(found['hits'])<count or count>=total or threshold_exhausted:break
            count=min(total,count*4)
        hits=[self.idx.hit(h['sid'],h['local_start'],h['squared_distance'],include_values) for h in chosen]
        for i,h in enumerate(hits):h['rank']=i+1
        return dict(hits=hits,certified=True,metric='z-normalized Euclidean',selection='greedy nearest remaining non-overlapping window',
            strategy='adaptive certified prefix',min_separation=separation,query_length=m,requested_k=k,exhausted=len(hits)<k,
            max_distance=max_distance,exclusion_scope='same file, column, device; global starts across chunks',
            certification_scope='Complete sorted nearest prefix; arbitrary equal-distance ties; not maximum-weight interval packing',
            windows=total,total_ms=(time.perf_counter()-began)*1000,prefix_rounds=rounds,
            positions_verified=sum(r['positions'] for r in rounds),symbol_positions=sum(r['symbols'] for r in rounds),
            middle_expanded=None,long_expanded=None)

    def search_bounded(self,query,k=10,min_separation=None,max_distance=None,include_values=False,
                       initial_candidates=512,max_candidates=8192):
        """Only O(candidate cap + one block) query distances; never an N-distance cache."""
        began=time.perf_counter();model=self.idx.model;m=model.length
        separation=m if min_separation is None else int(min_separation)
        if k<1 or separation<m or not k<=initial_candidates<=max_candidates:raise ValueError('Invalid candidate limits')
        if max_distance is not None and (not np.isfinite(max_distance) or max_distance<0):raise ValueError('Invalid distance limit')
        limit=np.inf if max_distance is None else float(max_distance)**2
        z,_,_,_=model.query(query);feature=model.transform(query)
        lookup=np.maximum(np.maximum(model.left-1e-4-feature[:,None],feature[:,None]-model.right-1e-4),0.)**2
        # Routing is only an ordering hint; every remaining block is verified.
        hint=self.base.search(query,k=min(k,10),verification='none')
        preferred=hint['selected_long_blocks'];seen=set(preferred)
        blocks=preferred+[i for i in range(len(self.base.blocks)) if i not in seen]
        count=min(initial_candidates,self.base.meta['windows']);cap=min(max_candidates,self.base.meta['windows'])
        rounds=[];peak_payload=0;peak_block_distances=0
        while True:
            # Release prior-round candidates before rescanning; rejected distances
            # are not retained. Three fixed-width arrays store distance/sid/start.
            best_d=np.empty(0);best_s=np.empty(0,dtype=np.int64);best_p=np.empty(0,dtype=np.int64)
            verified=0;symbols=0;round_start=time.perf_counter()
            for bid in blocks:
                sid,a,b,offset=map(int,self.base.blocks[bid])
                cutoff=min(limit,float(best_d[-1])) if len(best_d)>=count else limit
                if len(best_d)>=count and cutoff==0:break
                codes=self.base.codes[offset:offset+b-a];symbols+=len(codes)
                positions=kernels.filter_codes(codes,lookup,model.pairs*2,cutoff)+a
                if not len(positions):continue
                distances=kernels.exact_cached(self.idx.values(sid),positions,z,self.base.stats[sid],cutoff=cutoff)
                verified+=len(positions);peak_block_distances=max(peak_block_distances,len(distances))
                good=np.flatnonzero(distances<=cutoff)
                if len(good)>count:
                    good=good[np.argpartition(distances[good],count-1)[:count]]
                d=np.r_[best_d,distances[good]];s=np.r_[best_s,np.full(len(good),sid,dtype=np.int64)];p=np.r_[best_p,positions[good]]
                order=np.lexsort((p,s,d))[:count]
                # Numeric distance/location arrays only; Python metadata is made
                # for final k hits. This counter is payload, not process RSS.
                payload=best_d.nbytes+best_s.nbytes+best_p.nbytes+positions.nbytes+distances.nbytes+good.nbytes+d.nbytes+s.nbytes+p.nbytes+order.nbytes+24*len(order)
                peak_payload=max(peak_payload,payload)
                best_d,best_s,best_p=d[order],s[order],p[order]
                del d,s,p,distances,positions,good,order
            blocked=Exclusions();chosen=[]
            for i in range(len(best_d)):
                sid=int(best_s[i]);p=int(best_p[i]);start=self.offsets[sid]+p;key=self.keys[sid]
                if blocked.contains(key,start):continue
                chosen.append((float(best_d[i]),sid,p));blocked.add(key,start,separation)
                if len(chosen)>=k:break
            rounds.append(dict(requested=count,returned=len(best_d),selected=len(chosen),positions=verified,
                symbols=symbols,ms=(time.perf_counter()-round_start)*1000))
            exhausted=len(best_d)<count or count>=self.base.meta['windows']
            if len(chosen)>=k or exhausted or count>=cap:break
            count=min(cap,count*4)
        complete=len(chosen)>=k or exhausted
        hits=[self.idx.hit(s,p,d,include_values) for d,s,p in chosen]
        for i,h in enumerate(hits):h['rank']=i+1
        return dict(hits=hits,certified=complete,complete=complete,exhausted=bool(exhausted and len(hits)<k),
            candidate_limit_reached=not complete,strategy='bounded redundant prefix',requested_k=k,min_separation=separation,
            metric='z-normalized Euclidean',query_length=m,max_distance=max_distance,
            initial_candidates=initial_candidates,max_candidates=max_candidates,prefix_rounds=rounds,
            positions_verified=sum(r['positions'] for r in rounds)+hint['positions_verified'],
            symbol_positions=sum(r['symbols'] for r in rounds)+hint['symbol_positions_checked'],
            max_retained_candidates=max(r['returned'] for r in rounds),peak_block_distance_count=peak_block_distances,
            candidate_payload_bound_bytes=24*max_candidates,tracked_numeric_peak_bytes=peak_payload,
            memory_note='Candidate payload and tracked NumPy buffers only; excludes existing index, normalization stats, Python/NumPy overhead and routing scratch. No full-corpus query-distance cache.',
            windows=self.base.meta['windows'],total_ms=(time.perf_counter()-began)*1000,
            certification_scope='Greedy non-overlapping nearest results in completed index; floating tolerance and arbitrary ties; capped incomplete queries are not certified')

    def search(self,query,k=10,min_separation=None,max_distance=None,include_values=False):
        began=time.perf_counter();model=self.idx.model;m=model.length
        if k<1:raise ValueError('k must be positive')
        separation=m if min_separation is None else int(min_separation)
        if separation<m:raise ValueError('Separation must be at least query length to forbid overlaps')
        if max_distance is not None and (not np.isfinite(max_distance) or max_distance<0):raise ValueError('Invalid distance limit')
        limit=np.inf if max_distance is None else float(max_distance)**2
        z,ql,qu,_=model.query(query);feature=model.transform(query)
        qc=np.array([np.searchsorted(model.bins[j],v,side='right') for j,v in enumerate(feature)])
        lookup=np.maximum(np.maximum(model.left-1e-4-feature[:,None],feature[:,None]-model.right-1e-4),0.)**2
        blocked=Exclusions();pending=[];exact=[];leaves={};batches=[];hits=[]
        stats=dict(long_expanded=0,middle_expanded=0,symbol_positions=0,positions_verified=0,
            excluded_before_verification=0,excluded_from_exact=0)
        def excluded(sid,p):return blocked.contains(self.keys[sid],self.offsets[sid]+int(p))
        def allowed(sid,positions):
            ranges=blocked.ranges.get(self.keys[sid])
            if not ranges:return np.ones(len(positions),dtype=bool)
            ranges=np.asarray(ranges,dtype=np.int64);global_starts=positions+self.offsets[sid]
            i=np.searchsorted(ranges[:,0],global_starts,side='right')-1
            return (i<0)|(global_starts>=ranges[np.maximum(i,0),1])
        def push_boxes(kind,ids,low,high):
            lbs=kernels.bounds(low,high,model,ql,qu)
            centers=(low.astype(float)+high.astype(float))/2
            for i,node in enumerate(ids):
                if lbs[i]<=limit+1e-7:
                    heapq.heappush(pending,(float(lbs[i]),float(np.sum((qc-centers[i])**2)),kind,int(node),0))
        def advance(batch,pos):
            sid,positions,distances=batches[batch]
            while pos<len(positions):
                p=int(positions[pos])
                if not excluded(sid,p):
                    heapq.heappush(exact,(float(distances[pos]),sid,p,batch,pos));return
                stats['excluded_from_exact']+=1;pos+=1
        push_boxes(0,range(len(self.long_lo)),self.long_lo,self.long_hi)
        # Unresolved bounds and verified distances share a best-first frontier.
        # Accept only when no unresolved region can improve the chosen distance.
        while (pending or exact) and len(hits)<k:
            while exact and excluded(exact[0][1],exact[0][2]):
                _,_,_,batch,pos=heapq.heappop(exact);stats['excluded_from_exact']+=1;advance(batch,pos+1)
            if exact and (not pending or exact[0][0]==0 or pending[0][0]>exact[0][0]+1e-7):
                distance,sid,p,batch,pos=heapq.heappop(exact)
                hit=self.idx.hit(sid,p,distance,include_values);hit['rank']=len(hits)+1;hits.append(hit)
                blocked.add(self.keys[sid],self.offsets[sid]+p,separation)
                advance(batch,pos+1);continue
            if not pending:break
            lb,heuristic,kind,node,offset=heapq.heappop(pending)
            if kind==0:
                stats['long_expanded']+=1;ids=self.long_children[node]
                push_boxes(1,ids,self.lo[ids],self.hi[ids]);continue
            sid,a,b,code_offset,_=self.middle[node]
            if kind==1:
                stats['middle_expanded']+=1;positions=np.arange(a,b,dtype=np.int64)
                keep=allowed(sid,positions)
                stats['excluded_before_verification']+=int((~keep).sum())
                positions=positions[keep]
                if not len(positions):continue
                codes=self.base.codes[code_offset:code_offset+b-a][keep]
                lower=kernels.symbol_lower(codes,lookup,model.pairs*2);stats['symbol_positions']+=len(codes)
                order=np.argsort(lower,kind='stable');positions=positions[order];lower=lower[order]
                leaves[node]=(positions,lower)
                if lower[0]<=limit+1e-7:heapq.heappush(pending,(float(lower[0]),heuristic,2,node,0))
                continue
            positions,lower=leaves[node];stop=min(len(positions),offset+self.batch_size)
            selected=positions[offset:stop];keep=allowed(sid,selected)
            stats['excluded_before_verification']+=int((~keep).sum());selected=selected[keep]
            if stop<len(positions) and lower[stop]<=limit+1e-7:
                heapq.heappush(pending,(float(lower[stop]),heuristic,2,node,stop))
            if not len(selected):continue
            distances=kernels.exact_cached(self.idx.values(sid),selected,z,self.base.stats[sid])
            stats['positions_verified']+=len(selected)
            good=np.flatnonzero(distances<=limit);order=good[np.argsort(distances[good],kind='stable')]
            if len(order):
                batch=len(batches);batches.append((sid,selected[order],distances[order]));advance(batch,0)
        return dict(hits=hits,certified=True,metric='z-normalized Euclidean',selection='greedy nearest remaining non-overlapping window',
            min_separation=separation,query_length=m,requested_k=k,exhausted=len(hits)<k,max_distance=max_distance,
            exclusion_scope='same source file, column and device; global sample starts across internal chunks',
            certification_scope='Completed index, finite-precision conservative bounds, each greedy round; arbitrary equal-distance ties, not maximum-weight interval packing',
            windows=self.base.meta['windows'],total_ms=(time.perf_counter()-began)*1000,**stats)

    def close(self):self.idx.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',default='../results/cascade_index')
    ap.add_argument('--query-json',required=True);ap.add_argument('--k',type=int,default=10)
    ap.add_argument('--min-separation',type=int);ap.add_argument('--max-distance',type=float);ap.add_argument('--output')
    ap.add_argument('--strategy',choices=('hierarchy','prefix','bounded'),default='bounded')
    ap.add_argument('--initial-candidates',type=int,default=512);ap.add_argument('--max-candidates',type=int,default=8192)
    args=ap.parse_args();q=json.loads(Path(args.query_json).read_text(encoding='utf-8'));idx=DiverseIndex(args.index)
    try:
        method={'prefix':idx.search_prefix,'hierarchy':idx.search,'bounded':idx.search_bounded}[args.strategy]
        extra=dict(initial_candidates=args.initial_candidates,max_candidates=args.max_candidates) if args.strategy=='bounded' else (dict(initial_candidates=args.initial_candidates) if args.strategy=='prefix' else {})
        result=method(q if isinstance(q,list) else q['values'],k=args.k,min_separation=args.min_separation,
            max_distance=args.max_distance,include_values=True,**extra)
        text=json.dumps(result,ensure_ascii=False,indent=2)
        if args.output:
            out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(text,encoding='utf-8')
        print(text)
    finally:idx.close()

if __name__=='__main__':main()
