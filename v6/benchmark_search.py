"""Matched V6 before/after retrieval benchmark; frozen server weights, no training."""
import argparse
import functools
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from .inference import Retriever
from .index import Library, Candidates, diverse
from .data import Windows
from .common import write_json, environment, sha256


def import_snapshot(name, path):
    spec=importlib.util.spec_from_file_location('v6.'+name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class ExhaustiveLibrary(Library):
    """Audit every eligible stored vector in bounded batches; no tree pruning."""
    def search(self, query, channel='learned', k=10, sid=None, group=None, query_start=None,
               query_sid=None, leaf_budget=0, time_budget_ms=0, exclude_span=None,
               capacity=None, candidate_filter=None, **kwargs):
        q=np.asarray(query,dtype=np.float64)
        capacity=max(k,int(capacity or k*self.c['index']['oversample']))
        best=Candidates(capacity); length=self.c['length']
        span=exclude_span or length+max(self.c['horizons'])
        excluded=sid if query_sid is None else query_sid
        for i,shard in enumerate(self.meta['shards']):
            if sid is not None and sid!=shard['sid']: continue
            if group is not None and group!=shard['group']: continue
            sf=f'{channel}/starts.npy' if self.meta.get('layout')=='spatial' else 'starts.npy'
            starts=self.array(i,sf); vectors=self.array(i,f'{channel}/vectors.npy')
            for a in range(0,len(starts),4096):
                positions=starts[a:a+4096]
                allowed=np.ones(len(positions),dtype=bool)
                if query_start is not None and shard['sid']==excluded:
                    allowed &= np.abs(positions-query_start)>=span
                values=np.asarray(vectors[a:a+4096],dtype=np.float64)
                ds=np.sum((values-q)**2,axis=1)
                best.add(ds[allowed],shard['sid'],positions[allowed])
        hits=best.hits()
        if candidate_filter is not None: hits=candidate_filter(hits)
        selected=diverse(hits,k,length)
        return selected,dict(certified=True,complete_topk=len(selected)==k)


def identity(hits):
    return [(hit['sid'],hit['start']) for hit in hits]


def recall(actual, reference):
    return len(set(actual)&set(reference))/len(reference) if reference else float(not actual)


def benchmark(args):
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    old_index=import_snapshot('_old_search_index',Path(args.baseline_dir)/'baseline_index.py')
    old_inference=import_snapshot('_old_search_inference',Path(args.baseline_dir)/'baseline_inference.py')
    old_inference.Library=old_index.Library
    engines={'before':old_inference.Retriever(args.store,args.checkpoint,args.index)}
    for backend in ('numpy','native'):
        engine=Retriever(args.store,args.checkpoint,args.index)
        engine.library.search=functools.partial(engine.library.search,candidate_backend=backend)
        engines[backend]=engine
    data=Windows(args.store,engines['before'].c,args.split)
    rng=np.random.default_rng(args.seed)
    ids=rng.choice(len(data),min(args.queries,len(data)),replace=False)
    audit=set(int(qi) for qi in ids[:args.audit_queries])
    oracle=ExhaustiveLibrary(args.index)
    rows=[]
    try:
        for number,qi in enumerate(ids):
            sid,start=data.refs[int(qi)]
            # Read only history; benchmark does not inspect held-out query futures.
            x=data.store.window(sid,start,data.c['length'])
            for channel in args.channels:
                references={}; forecasts={}
                for budget in args.leaf_budgets:
                    for name,engine in engines.items():
                        out=engine.retrieve(x,sid,start,channel,budget)
                        if name=='before':
                            references[budget]=out['hits']; forecasts[budget]=out['prediction']
                oracle_ids=None
                if int(qi) in audit:
                    engine=engines['before']; saved=engine.library
                    try:
                        engine.library=oracle
                        oracle_ids=identity(engine.retrieve(x,sid,start,channel,0)['hits'])
                    finally: engine.library=saved
                schedule=[(name,budget,repeat) for repeat in range(args.repeats)
                          for budget in args.leaf_budgets for name in engines]
                rng.shuffle(schedule)
                for name,budget,repeat in schedule:
                    out=engines[name].retrieve(x,sid,start,channel,budget)
                    actual=identity(out['hits']); ref=identity(references[budget])
                    full=identity(references[0]) if 0 in references else None
                    rows.append(dict(query=int(qi),sid=sid,start=start,channel=channel,budget=budget,
                        method=name,repeat=repeat, same_order=actual==ref,
                        recall_same_budget=recall(actual,ref),
                        recall_full=recall(actual,full) if full is not None else None,
                        oracle_recall=recall(actual,oracle_ids) if oracle_ids is not None else None,
                        prediction_max_abs_difference=float(np.max(np.abs(out['prediction']-forecasts[budget]))),
                        **out['stats']))
                print(dict(query=number+1,total=len(ids),channel=channel),flush=True)
    finally:
        for engine in engines.values(): engine.close()
        data.store.close(); oracle.close()
    summary=[]
    for channel in args.channels:
        for budget in args.leaf_budgets:
            for method in engines:
                rr=[r for r in rows if (r['channel'],r['budget'],r['method'])==(channel,budget,method)]
                times=np.array([r['total_ms'] for r in rr]); audits=[r['oracle_recall'] for r in rr if r['oracle_recall'] is not None]
                summary.append(dict(channel=channel,budget=budget,method=method,queries=len(ids),measurements=len(rr),
                    p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),max_ms=float(times.max()),
                    within100ms=float(np.mean(times<=100)),search_p50_ms=float(np.median([r['search_ms'] for r in rr])),
                    exact_order_fraction=float(np.mean([r['same_order'] for r in rr])),
                    recall_same_budget=float(np.mean([r['recall_same_budget'] for r in rr])),
                    min_recall_same_budget=float(min(r['recall_same_budget'] for r in rr)),
                    recall_full=float(np.mean([r['recall_full'] for r in rr])) if 0 in args.leaf_budgets else None,
                    oracle_recall=float(np.mean(audits)) if audits else None,
                    max_prediction_difference=max(r['prediction_max_abs_difference'] for r in rr),
                    complete_topk_fraction=float(np.mean([r['complete_topk'] for r in rr]))))
    result=dict(config=vars(args),environment=environment(),threads=torch.get_num_threads(),
        windows=engines['before'].library.meta['windows'],checkpoint_sha256=sha256(args.checkpoint),
        index_sha256=sha256(Path(args.index)/'manifest.json'),
        baseline_commit=(Path(args.baseline_dir)/'baseline_commit.txt').read_text().strip(),
        audit_queries=len(audit),summary=summary,rows=rows,
        contract='Frozen weights/index; unchanged capacity, gates and budgets; equally warmed methods; randomized interleaved repeats. '
        'Includes history encoding, search, history gate and candidate future fetch. Startup excluded. '
        'Recall relative to same-budget baseline is implementation parity, not proof of full-library recall. '
        'Exhaustive oracle audits all eligible embeddings but retains only configured TopM before history gate/diversity.')
    write_json(output/'benchmark.json',result)
    report(result,output)
    return summary


def report(result,output):
    lines=['# V6 检索优化同机对照','',f"基线提交：`{result['baseline_commit']}`；索引窗口数：{result['windows']:,}。",'',
        '模型与索引冻结，不训练、不改门槛、不减少候选池。耗时包含编码、检索、历史门槛及候选未来读取；排除启动成本。','',
        '|通道|叶预算（0=完整）|实现|P50 ms|P95 ms|最大 ms|≤100ms|同预算位置召回|对完整搜索召回|暴力核验召回|返回完整K比例|预测最大差异|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['summary']:
        oracle='未测' if r['oracle_recall'] is None else f"{r['oracle_recall']:.2%}"
        full='未测' if r['recall_full'] is None else f"{r['recall_full']:.2%}"
        lines.append(f"|{r['channel']}|{r['budget']}|{r['method']}|{r['p50_ms']:.2f}|{r['p95_ms']:.2f}|{r['max_ms']:.2f}|{r['within100ms']:.2%}|{r['recall_same_budget']:.2%}|{full}|{oracle}|{r['complete_topk_fraction']:.2%}|{r['max_prediction_difference']:.3g}|")
    lines += ['',f"每项查询数：{result['summary'][0]['queries']}；每查询重复 {result['config']['repeats']} 次；暴力审计 {result['audit_queries']} 条查询。",'',
        '同预算召回衡量新旧实现是否一致。128 叶是近似预算，不能用同预算的 100% 取代相对完整搜索的召回。',
        '完整搜索与暴力审计均遵守现有 TopM 后历史门槛和非重叠筛选语义；不足 K 个时不保证全库满足门槛的非重叠 TopK 已找齐。',
        '本报告衡量检索实现的速度及输出保持情况，不证明未来相似度或预测质量提升。', '',
        '[完整逐查询数据](benchmark.json)']
    (Path(output)/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    channels=result['config']['channels']; budgets=result['config']['leaf_budgets']
    fig,axes=plt.subplots(1,len(channels),figsize=(6*len(channels),4),squeeze=False)
    for ax,channel in zip(axes[0],channels):
        x=np.arange(len(budgets)); width=.24
        for offset,method in enumerate(('before','numpy','native')):
            selected=[next(r for r in result['summary'] if (r['channel'],r['budget'],r['method'])==(channel,b,method)) for b in budgets]
            ax.bar(x+(offset-1)*width,[r['p95_ms'] for r in selected],width,label=method)
        ax.axhline(100,color='grey',linestyle=':'); ax.set_xticks(x,[str(b) for b in budgets])
        ax.set_xlabel('Leaf budget (0 = full search)'); ax.set_ylabel('End-to-end P95 (ms)')
        ax.set_title(channel); ax.legend(); ax.grid(axis='y',alpha=.2)
    fig.tight_layout(); fig.savefig(Path(output)/'latency.png',dpi=170); plt.close(fig)
    with (Path(output)/'REPORT.md').open('a',encoding='utf-8') as f:
        f.write('\n![端到端延迟对照](latency.png)\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('store','checkpoint','index','baseline-dir','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--queries',type=int,default=64); p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--audit-queries',type=int,default=8); p.add_argument('--split',choices=['validation','test'],default='test')
    p.add_argument('--channels',nargs='+',default=['learned','joint'])
    p.add_argument('--leaf-budgets',nargs='+',type=int,default=[0,128])
    p.add_argument('--seed',type=int,default=20260918); p.add_argument('--threads',type=int,default=1)
    a=p.parse_args(); torch.set_num_threads(a.threads)
    if min(a.queries,a.repeats,a.threads)<1 or min(a.leaf_budgets)<0 or a.audit_queries<0:
        p.error('Invalid benchmark limits')
    print(json.dumps(benchmark(a),indent=2))
