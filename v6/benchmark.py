"""Paired validation queries on two indices, without training or future labels."""
import argparse
import importlib.util
import time
import numpy as np
import torch
from .inference import Retriever
from .data import Windows
from .index import Library
from .common import write_json, environment, sha256


def benchmark(store, checkpoint, old_index, new_index, output, queries=16, baseline_module=None, repeats=2):
    old_class = Library
    if baseline_module:
        # Explicit local source under review, never a downloaded checkpoint/pickle.
        spec = importlib.util.spec_from_file_location('v6._benchmark_baseline', baseline_module)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        old_class = module.Library
    r = Retriever(store, checkpoint, old_index)
    data = Windows(store, r.c, 'validation')
    ids = np.random.default_rng(r.c['seed']).choice(len(data), min(queries,len(data)), replace=False)
    methods = {'before':old_class(old_index), 'cached':Library(old_index),
               'spatial_full':Library(new_index), 'spatial_128':Library(new_index)}
    result = []; expected = {}; rng=np.random.default_rng(r.c['seed'])
    try:
        # Prime every measured query for every method equally before timing.
        for name, library in methods.items():
            r.library = library
            for qi in ids:
                b = data[int(qi)]
                warm=r.retrieve(b['x'],int(b['sid']),int(b['start']),leaf_budget=128 if name.endswith('128') else 0)
                if name=='before': expected[int(qi)]=[(v['sid'],v['start']) for v in warm['hits']]
        schedule=[(name,int(qi),repeat) for repeat in range(repeats) for qi in ids for name in methods]
        rng.shuffle(schedule)
        for name,qi,repeat in schedule:
                r.library=methods[name]; b=data[qi]
                found = r.retrieve(b['x'],int(b['sid']),int(b['start']),leaf_budget=128 if name.endswith('128') else 0)
                identity = [(v['sid'],v['start']) for v in found['hits']]
                ref = expected[int(qi)]
                result.append(dict(method=name,query=int(qi),repeat=repeat,sid=int(b['sid']),
                    same_order=identity==ref, recall=len(set(identity)&set(ref))/max(len(ref),1), **found['stats']))
        summary = {}
        for name in methods:
            rr = [row for row in result if row['method']==name]
            ms = np.array([row['total_ms'] for row in rr])
            summary[name] = dict(queries=len(ids),measurements=len(rr),p50_ms=float(np.median(ms)),p95_ms=float(np.quantile(ms,.95)),
                max_ms=float(ms.max()), within100ms=float(np.mean(ms<=100)),
                mean_scored_fraction=float(np.mean([row['scored']/row['eligible'] for row in rr])),
                exact_order_fraction=float(np.mean([row['same_order'] for row in rr])),
                recall=float(np.mean([row['recall'] for row in rr])))
        write_json(output,dict(environment=environment(),checkpoint_sha256=sha256(checkpoint),
            split='validation',training=False,device='cpu',threads=torch.get_num_threads(),
            note='All measured queries warmed equally for all methods, then randomized interleaved repeats; warm service benchmark, NOT cold start; startup excluded.',
            summary=summary,rows=result))
        return summary
    finally:
        for library in methods.values(): library.close()
        r.close(); data.store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('store','checkpoint','old-index','new-index','output'): p.add_argument('--'+key,required=True)
    p.add_argument('--queries',type=int,default=16); p.add_argument('--baseline-module')
    p.add_argument('--threads',type=int,default=1)
    a=p.parse_args(); torch.set_num_threads(a.threads)
    print(benchmark(a.store,a.checkpoint,a.old_index,a.new_index,a.output,a.queries,a.baseline_module))
