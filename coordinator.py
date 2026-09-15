"""Safe worker routing: only proven lower bounds may skip an entire worker."""
import json,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import kernels
from model import LearnedShapeModel
from serve import model_id

def get_json(url,payload=None,timeout=300):
    data=None if payload is None else json.dumps(payload).encode('utf-8')
    req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)

class Coordinator:
    def __init__(self,urls,model_path):
        self.urls=[u.rstrip('/') for u in urls];self.model=LearnedShapeModel.load(model_path)
        self.info=[get_json(u+'/info') for u in self.urls]
        if len({i['index_id'] for i in self.info})!=1:raise ValueError('Workers must use the same completed index revision')
        if any(i['model_id']!=model_id(self.model) for i in self.info):raise ValueError('Worker model mismatch')
        owned=[s for i in self.info for s in i['owned_shards']];total=self.info[0]['total_shards']
        if len(owned)!=len(set(owned)) or sorted(owned)!=list(range(total)):
            raise ValueError('Workers must cover every shard exactly once; missing/duplicate coverage is rejected')
        self.lower=np.array([i['lower_codes'] for i in self.info],dtype=np.uint8)
        self.upper=np.array([i['upper_codes'] for i in self.info],dtype=np.uint8)

    def search(self,query,k=10,metric='ed',radius=8,budget_ms=None,parallel=4,include_values=False):
        began=time.perf_counter();_,ql,qu,_=self.model.query(query,'dtw' if metric=='robust' else metric,radius)
        bounds=kernels.bounds(self.lower,self.upper,self.model,ql,qu,metric)
        pending=sorted(range(len(bounds)),key=lambda i:bounds[i]);hits=[];responses=[];errors=[];skipped=[]
        def request(i):
            remaining=None if budget_ms is None else max(.01,budget_ms-(time.perf_counter()-began)*1000)
            data=dict(query=np.asarray(query).tolist(),k=k,metric=metric,radius=radius,
                budget_ms=remaining,include_values=include_values)
            return i,get_json(self.urls[i]+'/search',data,timeout=300 if budget_ms is None else max(1.,remaining/1000+.5))
        with ThreadPoolExecutor(max_workers=max(1,parallel)) as pool:
            while pending:
                cutoff=hits[k-1]['squared_distance'] if len(hits)>=k else np.inf
                retained=[]
                for i in pending:
                    if bounds[i]>cutoff+1e-7 or cutoff==0:skipped.append(i)
                    else:retained.append(i)
                pending=retained
                if not pending:break
                if budget_ms is not None and (time.perf_counter()-began)*1000>=budget_ms:break
                width=1 if not responses else max(1,parallel)
                batch=pending[:width];pending=pending[width:]
                futures=[(i,pool.submit(request,i)) for i in batch]
                for i,future in futures:
                    try:
                        _,r=future.result()
                        if r.get('index_id')!=self.info[i]['index_id']:raise ValueError('Worker index changed')
                        responses.append(dict(worker=i,certified=r['certified'],total_ms=r['total_ms']))
                        hits.extend(r['hits']);hits.sort(key=lambda h:(h['squared_distance'],h['sid'],h['local_start']));hits=hits[:k]
                    except Exception as exc:errors.append(dict(worker=i,error=str(exc)))
        zero=len(hits)>=k and hits[k-1]['squared_distance']==0
        certified=zero or (not pending and not errors and all(r['certified'] for r in responses))
        return dict(hits=hits,total_ms=(time.perf_counter()-began)*1000,certified=certified,
            workers_queried=len(responses)+len(errors),workers_skipped_by_bound=skipped,workers_pending=pending,
            responses=responses,errors=errors,windows=sum(i['windows'] for i in self.info),
            note='Local multi-worker tests validate protocol and coverage, not a deployed 10B-point cluster.')
