"""Approximate long-block routing, symbol screening, then exact point localization."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import kernels
from index import SearchIndex

class Cascade:
    @staticmethod
    def build(base,output,long_size=16384,mid_size=1024):
        if long_size<mid_size or long_size%mid_size:raise ValueError('Long size must be a multiple of middle size')
        out=Path(output);out.mkdir(parents=True,exist_ok=False);began=time.perf_counter()
        idx=SearchIndex(base);blocks=[];parts=[];codes_all=[];offset=0
        dims=np.arange(24).reshape(4,6);weights=8**np.arange(6,dtype=np.int64)
        for sid,s in enumerate(idx.manifest['series']):
            codes,_,_=kernels.encode(idx.values(sid),idx.model);codes_all.append(codes)
            for start in range(0,len(codes),long_size):
                end=min(len(codes),start+long_size);bid=len(blocks)
                blocks.append((sid,start,end,offset+start))
                quant=codes[start:end]//32
                for t,dd in enumerate(dims):
                    keys=np.unique((quant[:,dd].astype(np.int64)*weights).sum(axis=1)+t*8**6)
                    parts.append(np.c_[keys,np.full(len(keys),bid,dtype=np.int64)])
            offset+=len(codes)
        postings=np.concatenate(parts);order=np.argsort(postings[:,0],kind='stable');postings=postings[order]
        keys,starts,counts=np.unique(postings[:,0],return_index=True,return_counts=True)
        for name,value in dict(codes=np.concatenate(codes_all),blocks=np.array(blocks,dtype=np.int64),keys=keys,
            starts=starts,counts=counts,postings=postings[:,1].astype(np.int32),dims=dims).items():np.save(out/(name+'.npy'),value)
        meta=dict(base=str(Path(base).resolve()),manifest_hash=hashlib.sha256((Path(base)/'manifest.json').read_bytes()).hexdigest(),
            long_size=long_size,mid_size=mid_size,windows=offset,blocks=len(blocks),build_ms=(time.perf_counter()-began)*1000,
            method='4 six-symbol inverted tables, adjacent-bin probes, block vote, minimum symbol lower bound per middle block, exact ED')
        (out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');idx.close();return meta

    def __init__(self,path):
        self.path=Path(path);self.meta=json.loads((self.path/'manifest.json').read_text(encoding='utf-8'))
        base=Path(self.meta['base'])
        if hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest()!=self.meta['manifest_hash']:raise ValueError('Base index changed')
        self.idx=SearchIndex(base)
        for name in ('codes','blocks','keys','starts','counts','postings','dims'):setattr(self,name,np.load(self.path/(name+'.npy')))
        # Resident per-start moments preserve the exact centering arithmetic used
        # by the oracle; this trades 16 bytes/start for faster raw verification.
        began=time.perf_counter()
        self.stats=[kernels.local_stats(self.idx.values(sid),self.idx.model.length) for sid in range(len(self.idx.manifest['series']))]
        self.statistics_startup_ms=(time.perf_counter()-began)*1000

    def search(self,query,k=10,long_candidates=12,mid_candidates=16,probes=True,include_values=False,verification='scan'):
        if verification not in ('none','tree','scan'):raise ValueError('Unknown verification method')
        if min(k,long_candidates,mid_candidates)<1:raise ValueError('Positive candidate limits required')
        began=time.perf_counter();model=self.idx.model;z,ql,qu,_=model.query(query)
        feature=model.transform(query);code=np.array([np.searchsorted(model.bins[j],v,side='right') for j,v in enumerate(feature)])//32
        # Sparse dictionary: query work scales with touched postings, not all long blocks.
        votes={};weights=8**np.arange(6,dtype=np.int64);touched=0
        for t,dd in enumerate(self.dims):
            v=code[dd];key=int(v@weights+t*8**6);lookups=[(key,1.)]
            if probes:
                for j in range(6):
                    if v[j]>0:lookups.append((key-int(weights[j]),.25))
                    if v[j]<7:lookups.append((key+int(weights[j]),.25))
            table_votes={}
            for key,weight in lookups:
                p=int(np.searchsorted(self.keys,key))
                if p>=len(self.keys) or self.keys[p]!=key:continue
                a=int(self.starts[p]);count=int(self.counts[p]);touched+=count
                score=weight*np.log1p(len(self.blocks)/count)
                for bid in self.postings[a:a+count]:
                    bid=int(bid);table_votes[bid]=max(table_votes.get(bid,0.),score)
            for bid,score in table_votes.items():votes[bid]=votes.get(bid,0.)+score
        selected=sorted(votes,key=lambda b:(-votes[b],b))[:long_candidates]
        coarse_ms=(time.perf_counter()-began)*1000;middle=[];checked=0
        for bid in selected:
            sid,start,end,offset=map(int,self.blocks[bid]);codes=self.codes[offset:offset+end-start]
            lower=kernels.bounds(codes,codes,model,ql,qu);checked+=len(codes)
            for a in range(0,len(codes),self.meta['mid_size']):
                b=min(a+self.meta['mid_size'],len(codes))
                middle.append((float(lower[a:b].min()),sid,start+a,start+b,bid))
        middle.sort();chosen=middle[:mid_candidates];fine_ms=(time.perf_counter()-began)*1000-coarse_ms
        best=[];verified=0
        for lb,sid,a,b,bid in chosen:
            cutoff=best[-1][0] if len(best)>=k else np.inf
            if lb>cutoff+1e-7:continue
            positions=np.arange(a,b,dtype=np.int64)
            ds=kernels.exact_cached(self.idx.values(sid),positions,z,self.stats[sid],cutoff=cutoff);verified+=len(positions)
            good=np.flatnonzero(np.isfinite(ds)&(ds<=cutoff))
            order=good[np.argsort(ds[good],kind='stable')[:k]]
            best.extend((float(ds[i]),sid,int(positions[i])) for i in order);best.sort();best=best[:k]
        hits=[self.idx.hit(s,p,d,include_values) for d,s,p in best]
        result=dict(hits=hits,total_ms=(time.perf_counter()-began)*1000,coarse_ms=coarse_ms,middle_ms=fine_ms,
            metric='ed',certified=False,exact_distances_in_candidates=True,selected_long_blocks=selected,
            selected_middle_blocks=[dict(sid=s,start=a,end=b,long_id=bid) for _,s,a,b,bid in chosen],
            postings_touched=touched,symbol_positions_checked=checked,positions_verified=verified,
            windows=self.meta['windows'],warning='Approximate candidate routing; exact reranking does not certify global Top-k')
        if verification=='none':return result
        initial_ms=(time.perf_counter()-began)*1000
        if verification=='tree':
            exact=self.idx.search(query,k=k,seed_hits=hits,include_values=include_values)
            result.update(hits=exact['hits'],certified=exact['certified'],verification_positions=exact['positions_verified'],
                verification_nodes=exact['nodes_visited'])
        else:
            # A global certification pass. Scan symbols, then read raw points only
            # where the lower bound cannot rule out a missing top-k neighbor.
            delta=np.maximum(model.left-1e-4-feature[:,None],feature[:,None]-model.right-1e-4)
            lookup=np.maximum(delta,0.)**2;count=0;scanned=0
            for sid,start,end,offset in self.blocks:
                sid,start,end,offset=map(int,(sid,start,end,offset))
                cutoff=best[-1][0] if len(best)>=k else np.inf
                # k zero-distance results already prove an arbitrary top-k tie set.
                if cutoff==0:break
                codes=self.codes[offset:offset+end-start];scanned+=len(codes)
                positions=kernels.filter_codes(codes,lookup,model.pairs*2,cutoff)+start
                if not len(positions):continue
                ds=kernels.exact_cached(self.idx.values(sid),positions,z,self.stats[sid],cutoff=cutoff);count+=len(positions)
                good=np.flatnonzero(np.isfinite(ds)&(ds<=cutoff));order=good[np.argsort(ds[good],kind='stable')[:k]]
                best.extend((float(ds[i]),sid,int(positions[i])) for i in order)
                best=sorted(set(best))[:k]
            result.update(hits=[self.idx.hit(s,p,d,include_values) for d,s,p in best],certified=True,
                verification_positions=count,verification_symbol_positions=scanned)
        result.update(total_ms=(time.perf_counter()-began)*1000,initial_ms=initial_ms,
            verification_ms=(time.perf_counter()-began)*1000-initial_ms,verification=verification,
            warning='Certified for the completed index and z-normalized ED, conservative finite-precision bounds; arbitrary distance ties. No hard deadline.')
        return result

def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--base',default='results/validation_index');b.add_argument('--output',required=True)
    q=sub.add_parser('query');q.add_argument('--index',required=True);q.add_argument('--query-json',required=True)
    q.add_argument('--long-candidates',type=int,default=12);q.add_argument('--mid-candidates',type=int,default=16)
    q.add_argument('--verification',choices=('none','tree','scan'),default='scan')
    args=ap.parse_args()
    if args.command=='build':print(json.dumps(Cascade.build(args.base,args.output),indent=2));return
    data=json.loads(Path(args.query_json).read_text(encoding='utf-8'));obj=Cascade(args.index)
    print(json.dumps(obj.search(data if isinstance(data,list) else data['values'],long_candidates=args.long_candidates,
        mid_candidates=args.mid_candidates,include_values=True,verification=args.verification),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
