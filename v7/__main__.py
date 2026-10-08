"""Run python -m v7 --help from the repository root."""
import argparse
import json
from pathlib import Path


def serializable(value):
    if isinstance(value, dict): return {k:serializable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [serializable(v) for v in value]
    if hasattr(value, 'tolist'): return value.tolist()
    return value


def main():
    p = argparse.ArgumentParser(description='Moirai 2: A representation replacement / B V4 top20 reranking')
    subs = p.add_subparsers(dest='command', required=True)
    q = subs.add_parser('doctor'); q.add_argument('--config', required=True); q.add_argument('--device', default='cpu')
    q.add_argument('--download', action='store_true', help='Load pinned weights and run real model smoke inference')
    q = subs.add_parser('ingest'); q.add_argument('--config', required=True); q.add_argument('--output', required=True)
    q = subs.add_parser('train-a')
    for k in ('config','store','output'): q.add_argument('--'+k, required=True)
    q.add_argument('--device', default='cpu')
    q.add_argument('--no-feature-cache', action='store_true', help='Recompute frozen Moirai features in every training batch')
    q = subs.add_parser('index-a')
    for k in ('store','checkpoint','output'): q.add_argument('--'+k, required=True)
    q.add_argument('--device', default='cpu')
    q = subs.add_parser('index-b')
    for k in ('store','config','output'): q.add_argument('--'+k, required=True)
    for name in ('query-a','query-b'):
        q = subs.add_parser(name)
        for k in ('store','index','history','output'): q.add_argument('--'+k, required=True)
        q.add_argument('--sid', required=True, type=int); q.add_argument('--start', required=True, type=int)
        q.add_argument('--device', default='cpu')
        if name=='query-a':
            q.add_argument('--checkpoint', required=True)
            q.add_argument('--channel', choices=('learned','history','joint'), default='joint')
            q.add_argument('--leaf-budget', type=int, default=0)
    q = subs.add_parser('evaluate')
    for k in ('store','config','output'): q.add_argument('--'+k, required=True)
    for k in ('checkpoint-a','index-a','index-b','checkpoint-v6','index-v6'): q.add_argument('--'+k)
    q.add_argument('--split', choices=('validation','test'), default='test')
    q.add_argument('--device', default='cpu'); q.add_argument('--leaf-budget', type=int, default=0)
    args = vars(p.parse_args()); cmd = args.pop('command')
    from .common import config, write_json
    if cmd=='doctor':
        import torch
        import importlib.metadata
        c = config(args['config'])
        result = dict(torch=str(torch.__version__), cuda=torch.cuda.is_available(),
            uni2ts=importlib.metadata.version('uni2ts'), model=c['moirai'])
        if args['download']:
            from .moirai import MoiraiAdapter
            adapter = MoiraiAdapter.pretrained(c['moirai'],args['device'])
            x = torch.sin(torch.arange(c['length'],dtype=torch.float32)[None]/11).to(args['device'])
            result.update(hidden_shape=list(adapter.encode(x).shape),
                forecast_shape=list(adapter.forecast(x,max(c['horizons'])).shape), identity=adapter.identity)
        print(json.dumps(result,indent=2))
    elif cmd=='ingest':
        from v6.data import ingest
        config(args['config'])
        result = ingest(args['config'],args['output']); print(dict(points=result['total_points']))
    elif cmd=='train-a':
        from .train import fit
        print(fit(args['store'],args['config'],args['output'],args['device'],cache_features=not args['no_feature_cache']))
    elif cmd=='index-a':
        from .retrieval import build
        result=build(args['store'],args['checkpoint'],args['output'],args['device'])
        print(dict(windows=result['windows'],seconds=result['build_seconds']))
    elif cmd=='index-b':
        from .rerank import build
        result=build(args['store'],args['config'],args['output']); print(dict(series=len(result['series'])))
    elif cmd.startswith('query-'):
        import numpy as np
        if Path(args['output']).exists(): raise FileExistsError('Choose a fresh query output')
        if cmd=='query-a':
            from .retrieval import MoiraiRetriever
            engine=MoiraiRetriever(args['store'],args['checkpoint'],args['index'],args['device'])
            options=dict(channel=args['channel'],leaf_budget=args['leaf_budget'])
        else:
            from .rerank import V4MoiraiReranker
            engine=V4MoiraiReranker(args['store'],args['index'],args['device']); options={}
        try:
            result=engine.retrieve(np.load(args['history'],allow_pickle=False),args['sid'],args['start'],**options)
            write_json(args['output'],serializable(result)); print(json.dumps(result['stats'],indent=2))
        finally: engine.close()
    elif cmd=='evaluate':
        from .evaluate import evaluate
        args['config_path']=args.pop('config')
        evaluate(**args)


if __name__=='__main__': main()
