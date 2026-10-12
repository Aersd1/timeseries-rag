"""Evaluate frozen policies once on disjoint test windows absent from prior queries."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from v6.data import Windows,scale_floor
from v5.common import valid_ranges
from .common import fresh_dir,sha256,write_json
from .learned_rerank import LearnedMoiraiReranker
from .quality_gate import quality_features,gate_predict


def evaluate(root,source,name):
    folder=root/name; out=fresh_dir(folder/'extra_test')
    state=torch.load(folder/'quality_gate.pt',map_location='cpu',weights_only=True)
    gate_sha=sha256(folder/'quality_gate.pt'); policy_sha=sha256(root/'policy.json')
    engine=LearnedMoiraiReranker(source/name/'store',source/name/'train/best.pt',source/name/'index_a',candidates=100)
    original=Windows(source/name/'store',engine.c,'test'); previous=set(original.refs); c=engine.c
    refs=[]; span=c['length']+max(c['horizons'])
    for s in engine.store.series:
        for lo,end,step in valid_ranges(s,c['length'],max(c['horizons']),s['validation_end'],s['n'],span):
            refs.extend((s['sid'],int(p)) for p in range(lo,end,step) if (s['sid'],int(p)) not in previous)
    if not refs: raise ValueError('No independent remaining test episodes')
    assert not previous.intersection(refs)
    totals={m:[] for m in ['a50','a100','moirai_direct','a100_adaptive']}; records=[]
    try:
        for begin in range(0,len(refs),32):
            group=refs[begin:begin+32]
            histories=np.stack([engine.store.window(s,p,c['length']) for s,p in group])
            pools=engine.prepare_batch(histories,[s for s,p in group],[p for s,p in group])
            for (sid,start),x,prepared in zip(group,histories,pools):
                short={**prepared,'candidates':prepared['candidates'][:50],'mapped':prepared['mapped'][:50],
                    'future_scores':prepared['future_scores'][:50],'history_scores':prepared['history_scores'][:50]}
                r50=engine.rank(short,.2); r100=engine.rank(prepared,.2); diverse=engine.rank(prepared,.2,state['diversity_weight'])
                lookup={h['start']:i for i,h in enumerate(prepared['candidates'])}; ids=[lookup[h['start']] for h in diverse['hits']]
                series=engine.store.series[sid]
                feature=quality_features(x,scale_floor(series,c),prepared['quantiles'],prepared['quantile_levels'],
                    prepared['mapped'][ids],diverse['hits'],span,series['memory_std'])
                _,alpha=gate_predict(state,feature[None]); alpha=float(alpha[0])
                pred=dict(a50=r50['prediction'],a100=r100['prediction'],moirai_direct=prepared['direct_prediction'],
                    a100_adaptive=(1-alpha)*prepared['direct_prediction']+alpha*diverse['prediction'])
                # Only AFTER all predictions and choices, read the unknown answer for evaluation.
                y=engine.store.window(sid,start+c['length'],max(c['horizons'])).astype(float)
                scores={}
                for method,values in pred.items():
                    error=(np.asarray(values,dtype=float)-y)**2
                    score=float(np.mean([error[:h].mean()/max(series['memory_std'],1e-6)**2 for h in c['horizons']]))
                    totals[method].append(score); scores[method]=score
                for hit in prepared['candidates']:
                    assert hit['sid']==sid and hit['start']+span<=min(start,series['memory_end'])
                records.append(dict(sid=sid,start=start,analogue_alpha=alpha,scores=scores,
                    predictions={m:v.tolist() for m,v in pred.items()},selected=diverse['hits']))
            print('EXTRA',name,min(begin+32,len(refs)),'/',len(refs),flush=True)
        assert sha256(folder/'quality_gate.pt')==gate_sha and sha256(root/'policy.json')==policy_sha
        means={m:float(np.mean(v)) for m,v in totals.items()}
        result=dict(dataset=name,queries=len(refs),**means,gain_vs_a50_percent=100*(1-means['a100_adaptive']/means['a50']),
            excludes_previous_queries=True,gate_sha256=gate_sha,policy_sha256=policy_sha,
            query_overlap_violations=0,model_selection=False)
        write_json(out/'predictions.json',records); write_json(out/'summary.json',result)
        return result
    finally: engine.close(); original.store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',required=True)
    p.add_argument('--source',default='v7/runs/all_datasets_20261008'); args=p.parse_args()
    root,source=Path(args.root),Path(args.source); torch.set_num_threads(4)
    names=json.loads((root/'complete.json').read_text())['datasets']
    results=[evaluate(root,source,name) for name in names]
    write_json(root/'extra_test_summary.json',dict(complete=True,results=results,used_for_selection=False))
    print(json.dumps(results,indent=2))
