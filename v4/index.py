from __future__ import annotations
import heapq,json,time
from collections import OrderedDict
from pathlib import Path
import numpy as np
import kernels
from model import LearnedShapeModel
from tree import build_tree,save_tree,load_tree

RECORD=np.dtype([('sid','<i8'),('start','<i8'),('end','<i8'),('flat','u1')])

class IndexBuilder:
    """External coarse partitioning keeps feature memory bounded during ingest."""
    def __init__(self,path,model,shard_records=250000,leaf_size=256,resume=False):
        self.path=Path(path)
        if resume:
            self._resume(model,shard_records,leaf_size);return
        if (self.path/'manifest.json').exists() or (self.path/'raw.bin').exists():
            raise FileExistsError('Use a new build directory; existing data are never overwritten')
        self.path.mkdir(parents=True,exist_ok=True); model.save(self.path/'model')
        (self.path/'staging').mkdir(); self.model=model
        self.shard_records,self.leaf_size=shard_records,leaf_size
        self.stage_dtype=np.dtype([('code','u1',(model.dim,)),*RECORD.descr])
        self.raw=(self.path/'raw.bin').open('wb'); self.rowfile=(self.path/'rows.bin').open('wb')
        self.series=[];self.windows=0;self.points=0;self.unique_points=0;self.last_ends={}
        self.records=0;self.started=time.perf_counter();self.processed_files=[]
        self.checkpoint()

    def new_points(self,meta,n):
        key=(str(meta.get('source')),str(meta.get('column')),str(meta.get('device')))
        start=int(meta.get('start',0));last=self.last_ends.get(key,start)
        return max(0,start+n-max(start,last))

    def checkpoint(self):
        self.raw.flush();self.rowfile.flush()
        state=dict(raw_bytes=self.raw.tell(),row_bytes=self.rowfile.tell(),series=self.series,
            windows=self.windows,points=self.points,unique_points=self.unique_points,records=self.records,
            last_ends=[[*key,value] for key,value in self.last_ends.items()],
            processed_files=self.processed_files,shard_records=self.shard_records,leaf_size=self.leaf_size,
            stages={p.name:p.stat().st_size for p in (self.path/'staging').glob('*.bin')})
        temp=self.path/'checkpoint.tmp';temp.write_text(json.dumps(state,ensure_ascii=False),encoding='utf-8')
        temp.replace(self.path/'checkpoint.json')

    def _resume(self,model,shard_records,leaf_size):
        if (self.path/'manifest.json').exists():raise FileExistsError('This index is already complete')
        state=json.loads((self.path/'checkpoint.json').read_text(encoding='utf-8'))
        saved=LearnedShapeModel.load(self.path/'model')
        if model.length!=saved.length or not np.array_equal(model.freq,saved.freq) or not np.array_equal(model.bins,saved.bins):
            raise ValueError('Resume model must match the saved model')
        if (shard_records,leaf_size)!=(state['shard_records'],state['leaf_size']):raise ValueError('Resume layout mismatch')
        self.model=saved;self.shard_records=shard_records;self.leaf_size=leaf_size
        self.stage_dtype=np.dtype([('code','u1',(model.dim,)),*RECORD.descr])
        # Roll back only uncommitted append tails in this private build directory.
        for name,key in [('raw.bin','raw_bytes'),('rows.bin','row_bytes')]:
            path=self.path/name
            if path.stat().st_size<state[key]:raise ValueError('Checkpoint references missing raw data')
            with path.open('r+b') as f:f.truncate(state[key])
        for path in (self.path/'staging').glob('*.bin'):
            if path.name not in state['stages']:path.unlink()
            else:
                if path.stat().st_size<state['stages'][path.name]:raise ValueError('Missing staged records')
                with path.open('r+b') as f:f.truncate(state['stages'][path.name])
        for name in state['stages']:
            if not (self.path/'staging'/name).exists():raise ValueError('Missing staged partition')
        self.raw=(self.path/'raw.bin').open('ab');self.rowfile=(self.path/'rows.bin').open('ab')
        self.series=state['series'];self.windows=state['windows'];self.points=state['points']
        self.unique_points=state['unique_points'];self.records=state['records']
        self.last_ends={tuple(v[:3]):v[3] for v in state['last_ends']};self.processed_files=state['processed_files']
        self.started=time.perf_counter()

    def add(self,values,meta,rows=None):
        x=np.ascontiguousarray(values,dtype=float)
        if len(x)<self.model.length:return
        if not np.isfinite(x).all():raise ValueError('Only contiguous finite runs may be added')
        sid=len(self.series); info=dict(meta,offset=self.raw.tell()//8,n=len(x))
        if rows is not None:
            rows=np.asarray(rows,dtype=np.int64); difference=np.diff(rows)
            if len(rows)!=len(x):raise ValueError('Row mapping length mismatch')
            if len(difference)==0 or np.all(difference==difference[0]):
                info.update(row_start=int(rows[0]),row_stride=int(difference[0]) if len(difference) else 1)
            else:
                info['row_offset']=self.rowfile.tell()//8;rows.tofile(self.rowfile)
        self.unique_points+=self.new_points(meta,len(x))
        key=(str(meta.get('source')),str(meta.get('column')),str(meta.get('device')))
        self.last_ends[key]=max(self.last_ends.get(key,0),int(meta.get('start',0))+len(x))
        x.tofile(self.raw); self.series.append(info)
        codes,flat,_=kernels.encode(x,self.model)
        change=np.r_[True,np.any(codes[1:]!=codes[:-1],axis=1)|(flat[1:]!=flat[:-1])]
        starts=np.flatnonzero(change);ends=np.r_[starts[1:],len(codes)]
        records=np.empty(len(starts),dtype=self.stage_dtype)
        records['code']=codes[starts];records['sid']=sid;records['start']=starts
        records['end']=ends;records['flat']=flat[starts]
        # Shape partition, not source-file partition; later directory boxes can
        # rule out whole groups of shards before their leaves are opened.
        axes=np.arange(min(4,self.model.pairs))*2
        keys=((records['code'][:,axes].astype(np.uint16)>>6) << (2*np.arange(len(axes)))).sum(axis=1)
        order=np.argsort(keys,kind='stable');records=records[order];keys=keys[order]
        edges=np.r_[0,np.flatnonzero(keys[1:]!=keys[:-1])+1,len(keys)]
        for a,b in zip(edges[:-1],edges[1:]):
            with (self.path/'staging'/f'{int(keys[a]):03d}.bin').open('ab') as f:
                records[a:b].tofile(f)
        self.windows+=len(codes);self.points+=len(x);self.records+=len(records)

    def finish(self,scope=None):
        self.checkpoint()
        self.raw.close();self.rowfile.close(); shards=[]; root_l=[]; root_u=[]
        for stage in sorted((self.path/'staging').glob('*.bin')):
            with stage.open('rb') as f:
                while True:
                    batch=np.fromfile(f,dtype=self.stage_dtype,count=self.shard_records)
                    if not len(batch):break
                    shid=len(shards); dest=self.path/'shards'/f'{shid:06d}'
                    codes=np.ascontiguousarray(batch['code']); tree=build_tree(codes,leaf_size=self.leaf_size)
                    # Physical records follow leaf order for contiguous SSD access.
                    order=tree['order']; codes=codes[order]
                    rec=np.empty(len(batch),dtype=RECORD)
                    for name in RECORD.names:rec[name]=batch[name][order]
                    tree['order']=np.arange(len(batch),dtype=np.int64)
                    save_tree(dest,tree);np.save(dest/'codes.npy',codes);np.save(dest/'records.npy',rec)
                    weights=int(np.sum(rec['end']-rec['start']))
                    shards.append(dict(id=shid,records=len(batch),windows=weights,nodes=len(tree['lo'])))
                    root_l.append(tree['lo'][0]);root_u.append(tree['hi'][0])
        directory=build_tree(np.asarray(root_l),np.asarray(root_u),leaf_size=1) if shards else None
        if directory is None:raise ValueError('No searchable windows were indexed')
        save_tree(self.path/'directory',directory)
        manifest=dict(version=1,windows=self.windows,stored_points=self.points,unique_raw_points=self.unique_points,records=self.records,
            series=self.series,shards=shards,scope=scope or {},build_seconds=time.perf_counter()-self.started,
            complete=True,metric_contract='z-normalized ED or fixed-radius DTW; all valid starts retained')
        temp=self.path/'manifest.tmp';temp.write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf-8')
        temp.replace(self.path/'manifest.json')
        # Keep all staging until publication, so interrupted compaction can restart.
        for stage in (self.path/'staging').glob('*.bin'):stage.unlink()
        return manifest


class SearchIndex:
    def __init__(self,path,cache_shards=256,owned_shards=None):
        self.path=Path(path);self.model=LearnedShapeModel.load(self.path/'model')
        self.manifest=json.loads((self.path/'manifest.json').read_text(encoding='utf-8'))
        if self.manifest['version']!=1 or not self.manifest['complete']:raise ValueError('Incomplete or incompatible index')
        self.directory=load_tree(self.path/'directory');self.raw=np.memmap(self.path/'raw.bin',dtype='<f8',mode='r')
        self.rows=np.memmap(self.path/'rows.bin',dtype='<i8',mode='r') if (self.path/'rows.bin').stat().st_size else None
        self.cache=OrderedDict();self.cache_shards=cache_shards
        self.offsets=np.array([s['offset'] for s in self.manifest['series']],dtype=np.int64)
        self.owned_shards=list(range(len(self.manifest['shards']))) if owned_shards is None else sorted(set(owned_shards))
        if not self.owned_shards or self.owned_shards[0]<0 or self.owned_shards[-1]>=len(self.manifest['shards']):raise ValueError('Invalid shard ownership')
        mask=np.zeros(len(self.manifest['shards']),dtype=np.int64);mask[self.owned_shards]=1
        prefix=np.r_[0,np.cumsum(mask[self.directory['order']])]
        spans=self.directory['spans'];self.node_owned=(prefix[spans[:,1]]-prefix[spans[:,0]])>0
        self.owned_set=set(self.owned_shards)
        self.search_windows=sum(self.manifest['shards'][s]['windows'] for s in self.owned_shards)
        total=len(self.manifest['shards']);self.root_lo=np.empty((total,self.model.dim),dtype=np.uint8);self.root_hi=np.empty_like(self.root_lo)
        for node in np.flatnonzero(self.directory['children'][:,0]<0):
            a,b=self.directory['spans'][node];ids=self.directory['order'][a:b]
            self.root_lo[ids]=self.directory['lo'][node];self.root_hi[ids]=self.directory['hi'][node]

    def warm_metadata(self):
        for sid in self.owned_shards[:self.cache_shards]:self.shard(sid)

    def shard(self,sid):
        if sid in self.cache:
            self.cache.move_to_end(sid);return self.cache[sid]
        path=self.path/'shards'/f'{sid:06d}'; obj=load_tree(path)
        obj['codes']=np.load(path/'codes.npy',mmap_mode='r');obj['records']=np.load(path/'records.npy',mmap_mode='r')
        self.cache[sid]=obj
        if len(self.cache)>self.cache_shards:self.cache.popitem(last=False)
        return obj

    def values(self,sid):
        info=self.manifest['series'][sid];return self.raw[info['offset']:info['offset']+info['n']]

    def hit(self,sid,start,dist,include_values):
        info=self.manifest['series'][sid]; m=self.model.length
        result=dict(info,sid=int(sid),local_start=int(start),start=int(info.get('start',0))+int(start),
            squared_distance=float(dist),distance=float(np.sqrt(dist)))
        for key in ('offset','n','row_offset'):result.pop(key,None)
        if 'row_offset' in info:
            r=self.rows[info['row_offset']+start:info['row_offset']+start+m]
            result.update(source_row_start=int(r[0]),source_row_end=int(r[-1])+1)
        elif 'row_start' in info:
            result.update(source_row_start=info['row_start']+int(start)*info['row_stride'],
                source_row_end=info['row_start']+int(start+m-1)*info['row_stride']+1)
        if include_values:result['values']=self.values(sid)[start:start+m].tolist()
        return result

    def search(self,query,k=10,metric='ed',radius=8,time_budget_ms=None,include_values=False,seed_hits=None):
        began=time.perf_counter();model=self.model
        if k<1 or (time_budget_ms is not None and time_budget_ms<=0):raise ValueError('Invalid search limits')
        z,ql,qu,env=model.query(query,metric,radius)
        qf=model.transform(query);qc=np.array([np.searchsorted(model.bins[j],v,side='right') for j,v in enumerate(qf)],dtype=float)
        ds=np.full(k,np.inf);sids=np.full(k,-1,dtype=np.int64);positions=sids.copy()
        count=np.zeros(1,dtype=np.int64);constants=count.copy();stats=np.zeros(6,dtype=np.int64);owners={}
        # A cheap guided leaf supplies a valid initial upper bound. It never
        # excludes a region; the complete lower-bound traversal still follows.
        internal_seeds=False
        if metric=='ed' and seed_hits is None:
            internal_seeds=True
            tree=self.directory;node=0;owner=None
            while True:
                left,right=tree['children'][node]
                if left>=0:
                    children=[int(left),int(right)]
                    if owner is None:children=[c for c in children if self.node_owned[c]]
                    lower=kernels.bounds(tree['lo'][children],tree['hi'][children],model,ql,qu,metric)
                    centers=(tree['lo'][children].astype(float)+tree['hi'][children])/2
                    node=children[min(range(len(children)),key=lambda i:(lower[i],float(np.sum((qc-centers[i])**2))))]
                    continue
                a,b=tree['spans'][node]
                if owner is None:
                    candidates=[int(s) for s in tree['order'][a:b] if int(s) in self.owned_set]
                    owner=min(candidates,key=lambda s:float(np.sum((qc-(self.root_lo[s].astype(float)+self.root_hi[s])/2)**2)))
                    tree=self.shard(owner);node=0;continue
                rec=tree['records'][a:b];codes=tree['codes'][a:b]
                lower=kernels.bounds(codes,codes,model,ql,qu,metric)
                seed_hits=[]
                for r in np.argsort(lower)[:16]:
                    entry=rec[r]
                    for pos in range(int(entry['start']),min(int(entry['end']),int(entry['start'])+k)):
                        seed_hits.append(dict(sid=int(entry['sid']),local_start=pos,shard_id=owner))
                break
        if seed_hits:
            seeds=[];seen=set()
            for hit in seed_hits:
                if internal_seeds and time_budget_ms is not None and (time.perf_counter()-began)*1000>=min(20.,time_budget_ms*.25):break
                sid=int(hit['sid']);pos=int(hit['local_start'])
                if (sid,pos) in seen:continue
                if not 0<=sid<len(self.manifest['series']) or not 0<=pos<=self.manifest['series'][sid]['n']-model.length:
                    raise ValueError('Invalid seed position')
                owner=hit.get('shard_id')
                if owner is None and len(self.owned_shards)!=len(self.manifest['shards']):raise ValueError('Worker seeds require a verified shard_id')
                if owner is not None:
                    owner=int(owner)
                    if owner not in self.owned_set:raise ValueError('Seed is outside worker ownership')
                    if not internal_seeds:
                        rec=self.shard(owner)['records']
                        if not np.any((rec['sid']==sid)&(rec['start']<=pos)&(rec['end']>pos)):raise ValueError('Seed is not in the stated shard')
                owners[(sid,pos)]=owner
                seen.add((sid,pos))
                # Recompute: neither an agent nor a caller may supply an unsafe upper bound.
                values,counters=kernels.exact(self.values(sid),np.array([pos]),z,metric,radius,np.inf,env)
                stats[3]+=1;stats[4]+=int(counters[1]) if metric=='dtw' else 0
                seeds.append((float(values[0]),sid,pos))
            seeds.sort();seeds=seeds[:k];count[0]=len(seeds)
            for i,(distance,sid,pos) in enumerate(seeds):ds[i]=distance;sids[i]=sid;positions[i]=pos
        queue=[];opened=set();done=set();truncated=False;frontier=np.inf;directory_nodes=0
        def remaining():return None if time_budget_ms is None else time_budget_ms-(time.perf_counter()-began)*1000
        def threshold():return ds[k-1] if count[0]>=k else np.inf
        def push(node):
            if not self.node_owned[node]:return
            t=self.directory;lb=float(kernels.bounds(t['lo'][[node]],t['hi'][[node]],model,ql,qu,metric)[0])
            if lb>threshold()+1e-7:return
            center=(t['lo'][node].astype(float)+t['hi'][node])/2
            heapq.heappush(queue,(lb,float(np.sum((qc-center)**2)),int(node)))
        def process(shid,lower):
            nonlocal truncated,frontier,stats
            if shid in done:return
            rem=remaining()
            if rem is not None and rem<=0:truncated=True;frontier=min(frontier,lower);return
            tree=self.shard(shid);opened.add(shid)
            rem=remaining()
            if rem is not None and rem<=0:truncated=True;frontier=min(frontier,lower);return
            stop,local,pending=kernels.search_tree(self.raw,self.offsets,tree,model,ql,qu,z,metric,radius,
                ds,sids,positions,count,constants,rem,env)
            stats+=local
            for i in range(int(count[0])):
                key=(int(sids[i]),int(positions[i]))
                if owners.get(key) is None:owners[key]=shid
            if stop:truncated=True;frontier=min(frontier,lower,pending)
            else:done.add(shid)
        # Directory traversal is ordered by safe lower bound, with a learned-code
        # geometric tie-break; native kernels traverse each selected local tree.
        push(0)
        while queue and not truncated:
            if threshold()==0.:break
            rem=remaining()
            if rem is not None and rem<=0:truncated=True;break
            lower,_,node=heapq.heappop(queue)
            if lower>threshold()+1e-7:continue
            directory_nodes+=1;t=self.directory;left,right=t['children'][node]
            if left>=0:push(int(left));push(int(right))
            else:
                a,b=t['spans'][node]
                for shid in t['order'][a:b]:
                    if int(shid) in self.owned_set:process(int(shid),lower)
        if queue:frontier=min(frontier,queue[0][0])
        elapsed=(time.perf_counter()-began)*1000
        return dict(hits=[dict(self.hit(int(sids[i]),int(positions[i]),float(ds[i]),include_values),
            shard_id=owners.get((int(sids[i]),int(positions[i])))) for i in range(int(count[0]))],
            metric=metric,warping_radius=radius if metric=='dtw' else None,total_ms=elapsed,
            certified=not truncated,budget_exhausted=truncated,frontier_lower_bound=None if np.isinf(frontier) else frontier,
            windows=self.search_windows,records=sum(self.manifest['shards'][s]['records'] for s in self.owned_shards),shards=len(self.owned_shards),
            nodes_visited=int(stats[0])+directory_nodes,leaves_visited=int(stats[1]),symbol_records_checked=int(stats[2]),
            positions_verified=int(stats[3]),dtw_evaluations=int(stats[4]),constant_positions_covered=int(stats[5]),
            shards_opened=len(opened),positions_reduction=self.search_windows/max(1,int(stats[3])),
            certification_scope='Current completed index only; specified metric, finite precision with conservative bounds; any k distance ties.')

    def robust_search(self,query,k=10,radius=8,time_budget_ms=None,include_values=False):
        began=time.perf_counter()
        seed_budget=None if time_budget_ms is None else max(.01,time_budget_ms*.25)
        seed=self.search(query,k=max(10,k),metric='ed',time_budget_ms=seed_budget)
        remaining=None if time_budget_ms is None else max(.01,time_budget_ms-(time.perf_counter()-began)*1000)
        result=self.search(query,k=k,metric='dtw',radius=radius,time_budget_ms=remaining,
            include_values=include_values,seed_hits=seed['hits'])
        result.update(total_ms=(time.perf_counter()-began)*1000,seed_stage_ms=seed['total_ms'],
            seed_positions_verified=seed['positions_verified'],pipeline='ED candidate seeds -> verified DTW upper bound -> complete DTW tree search')
        return result

    def search_reference(self,query,k=10,metric='ed',radius=8,time_budget_ms=None,include_values=False):
        began=time.perf_counter(); model=self.model
        if k<1 or (time_budget_ms is not None and time_budget_ms<=0):raise ValueError('Invalid search limits')
        zq,ql,qu,envelope=model.query(query,metric,radius)
        qfeature=model.transform(query)
        qcode=np.array([np.searchsorted(model.bins[j],v,side='right') for j,v in enumerate(qfeature)],dtype=float)
        best=[];visited=set();queue=[];truncated=False;unfinished=np.inf;constant_count=0
        stats=dict(nodes_visited=0,leaves_visited=0,symbol_records_checked=0,positions_verified=0,
            dtw_evaluations=0,constant_positions_covered=0,shards_opened=0)
        opened=set()
        def expired():return time_budget_ms is not None and (time.perf_counter()-began)*1000>=time_budget_ms
        def threshold():return best[-1][0] if len(best)>=k else np.inf
        def box(tree,ids):
            return kernels.bounds(tree['lo'][ids],tree['hi'][ids],model,ql,qu,metric)
        def push(kind,sid,node,tree):
            lower=float(box(tree,[node])[0])
            if lower>threshold()+1e-7:return
            center=(tree['lo'][node].astype(float)+tree['hi'][node])/2
            heuristic=float(np.sum((qcode-center)**2))
            heapq.heappush(queue,(lower,heuristic,kind,sid,node))
        def refine(sid,node,tree,lower):
            nonlocal best,constant_count,unfinished
            if (sid,node) in visited:return True
            visited.add((sid,node));stats['leaves_visited']+=1
            a,b=tree['spans'][node]; codes=tree['codes'][a:b]
            lb=kernels.bounds(codes,codes,model,ql,qu,metric);stats['symbol_records_checked']+=len(codes)
            order=np.argsort(lb,kind='stable')
            for r in order:
                if lb[r]>threshold()+1e-7:break
                rec=tree['records'][a+r];source=int(rec['sid']);start=int(rec['start']);end=int(rec['end'])
                if rec['flat']:
                    stats['constant_positions_covered']+=end-start
                    if constant_count>=k:continue
                    dist=float(zq@zq)
                    take=min(k-constant_count,end-start)
                    best.extend((dist,source,start+j) for j in range(take));constant_count+=take
                    best.sort();best=best[:k];continue
                for first in range(start,end,128):
                    if expired():unfinished=min(unfinished,lower);return False
                    positions=np.arange(first,min(first+128,end),dtype=np.int64)
                    cutoff=threshold()
                    distances,count=kernels.exact(self.values(source),positions,zq,metric,radius,cutoff,envelope)
                    stats['positions_verified']+=len(positions)
                    if metric=='dtw':stats['dtw_evaluations']+=int(count[1])
                    best.extend((float(d),source,int(pos)) for pos,d in zip(positions,distances) if np.isfinite(d) and d<=cutoff)
                    best.sort();best=best[:k]
            return True
        # Guided traversal obtains an upper bound. It never certifies an exclusion.
        tree=self.directory;kind=0;sid=-1;node=0
        while True:
            left,right=tree['children'][node]
            if left>=0:
                ids=[int(left),int(right)];lbs=box(tree,ids)
                node=min(zip(ids,lbs),key=lambda p:(p[1],np.sum((qcode-(tree['lo'][p[0]].astype(float)+tree['hi'][p[0]])/2)**2)))[0]
            elif kind==0:
                a,b=tree['spans'][node]
                sid=int(tree['order'][a]);tree=self.shard(sid);opened.add(sid);kind=1;node=0
            else:
                if not refine(sid,node,tree,float(box(tree,[node])[0])):truncated=True
                break
            if expired():truncated=True;unfinished=0.;break
        push(0,-1,0,self.directory)
        while queue and not truncated:
            if threshold()==0.:break  # Any k zero-distance ties are globally optimal.
            if expired():truncated=True;break
            lower,_,kind,sid,node=heapq.heappop(queue)
            if lower>threshold()+1e-7:continue
            tree=self.directory if kind==0 else self.shard(sid)
            if kind:opened.add(sid)
            stats['nodes_visited']+=1
            left,right=tree['children'][node]
            if left>=0:
                push(kind,sid,int(left),tree);push(kind,sid,int(right),tree)
            elif kind==0:
                a,b=tree['spans'][node]
                for sh in tree['order'][a:b]:
                    sh=int(sh);obj=self.shard(sh);opened.add(sh);push(1,sh,0,obj)
            elif not refine(sid,node,tree,lower):
                truncated=True
        elapsed=(time.perf_counter()-began)*1000;stats['shards_opened']=len(opened)
        pending=min(queue[0][0] if queue else np.inf,unfinished)
        return dict(hits=[self.hit(s,p,d,include_values) for d,s,p in best],metric=metric,warping_radius=radius if metric=='dtw' else None,
            total_ms=elapsed,certified=not truncated,budget_exhausted=truncated,
            frontier_lower_bound=None if np.isinf(pending) else pending,
            windows=self.manifest['windows'],records=self.manifest['records'],shards=len(self.manifest['shards']),
            positions_reduction=self.manifest['windows']/max(1,stats['positions_verified']),**stats,
            certification_scope='Current completed index only; specified metric, finite precision with conservative bounds; any k distance ties.')

    def exhaustive(self,query,k=10,metric='ed',radius=8):
        if len(self.owned_shards)!=len(self.manifest['shards']):raise ValueError('Full oracle requires all shards')
        began=time.perf_counter();z,_,_,env=self.model.query(query,metric,radius);best=[];count=0
        for sid,info in enumerate(self.manifest['series']):
            n=info['n']-self.model.length+1
            for start in range(0,n,1024):
                ids=np.arange(start,min(start+1024,n),dtype=np.int64)
                cutoff=best[-1][0] if len(best)>=k else np.inf
                ds,_=kernels.exact(self.values(sid),ids,z,metric,radius,cutoff,env)
                take=min(k,len(ids));ix=np.argpartition(ds,take-1)[:take]
                best.extend((float(ds[j]),sid,int(ids[j])) for j in ix);best.sort();best=best[:k];count+=len(ids)
        return dict(hits=[self.hit(s,p,d,False) for d,s,p in best],total_ms=(time.perf_counter()-began)*1000,
            positions_verified=count,certified=True)

    def close(self):
        for tree in [self.directory,*self.cache.values()]:
            for value in tree.values():
                if isinstance(value,np.memmap):value._mmap.close()
        self.cache.clear();self.raw._mmap.close()
        if self.rows is not None:self.rows._mmap.close()

    def prepare_fft(self):
        self.fft_stats=[kernels.window_stats(self.values(i),self.model.length) for i in range(len(self.manifest['series']))]

    def fft_exhaustive(self,query,k=10):
        if len(self.owned_shards)!=len(self.manifest['shards']):raise ValueError('Full oracle requires all shards')
        from scipy.signal import fftconvolve
        if not hasattr(self,'fft_stats'):self.prepare_fft()
        began=time.perf_counter();z,_,_,_=self.model.query(query);m=len(z);energy=float(z@z);best=[]
        for sid,(mean,sd) in enumerate(self.fft_stats):
            x=self.values(sid);y=x-x[0]
            dot=fftconvolve(y,z[::-1],mode='valid')-mean*z.sum()
            ds=np.maximum(m+energy-2*dot/np.where(sd>1e-10,sd,1),0);ds[sd<=1e-10]=energy
            take=min(len(ds),max(32,k*4));pos=np.argpartition(ds,take-1)[:take]
            exact,_=kernels.exact(x,pos,z)
            best.extend((float(d),sid,int(p)) for p,d in zip(pos,exact));best.sort();best=best[:k]
        return dict(hits=[self.hit(s,p,d,False) for d,s,p in best],total_ms=(time.perf_counter()-began)*1000)
