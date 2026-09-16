"""Run python -m v6 --help from the repository root."""
import argparse
import json
from pathlib import Path
from .common import config


def main():
    p = argparse.ArgumentParser(description='V6 future-belief hierarchical retrieval')
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('doctor', 'ingest', 'estimate'):
        q = sub.add_parser(name); q.add_argument('--config', required=True)
        if name == 'ingest': q.add_argument('--output', required=True)
        if name == 'estimate': q.add_argument('--points', type=int, default=10_000_000_000)
    train = sub.add_parser('train')
    train.add_argument('--config', required=True); train.add_argument('--store', required=True)
    train.add_argument('--stage', choices=('prior','encoder'), required=True)
    train.add_argument('--output', required=True); train.add_argument('--prior')
    train.add_argument('--device', default='cpu'); train.add_argument('--resume', action='store_true')
    index = sub.add_parser('index')
    for key in ('store','checkpoint','output'): index.add_argument('--'+key, required=True)
    index.add_argument('--device', default='cpu')
    ev = sub.add_parser('evaluate')
    for key in ('store','checkpoint','index','output'): ev.add_argument('--'+key, required=True)
    ev.add_argument('--device', default='cpu'); ev.add_argument('--split', choices=('validation','test'), default='test')
    ev.add_argument('--leaf-budgets', type=int, nargs='+'); ev.add_argument('--time-budget-ms', type=float)
    ev.add_argument('--oversample', type=int)
    an = sub.add_parser('analyze'); an.add_argument('--results', required=True)
    an.add_argument('--prior-training'); an.add_argument('--encoder-training')
    compare = sub.add_parser('compare')
    for key in ('first','second','output'): compare.add_argument('--'+key, required=True)
    query = sub.add_parser('query')
    for key in ('store','checkpoint','index','history','output'): query.add_argument('--'+key, required=True)
    query.add_argument('--sid', type=int, required=True); query.add_argument('--start', type=int)
    query.add_argument('--device', default='cpu'); query.add_argument('--leaf-budget', type=int, default=0)
    args = vars(p.parse_args()); command = args.pop('command')
    if command == 'doctor':
        import glob
        import torch
        c = config(args['config'])
        print(dict(torch=str(torch.__version__), cuda=torch.cuda.is_available()))
        for source in c['data']['sources']:
            matched = len(glob.glob(source['glob'], recursive=True))
            print(dict(group=source.get('group'), csv_files=matched))
            if not matched: raise SystemExit('Edit CSV paths before ingest (or use an existing compatible V5 store).')
    elif command == 'estimate':
        c = config(args['config']); m = c['model']; n = args['points']//c['index']['stride']
        dims = dict(learned=m['dim'], history=m['history_bins'], belief=m['components']*(1+2*m['belief_bins']))
        d = sum(dims[ch] for ch in c['index']['channels'])
        print(json.dumps(dict(assumption='All points eligible; excludes raw CSV and model workspace', windows=n,
            vector_GB=n*d*4/1e9, starts_GB=n*8/1e9,
            approximate_boxes_GB=n/c['index']['leaf_size']*8*d*c['index']['fanout']/(c['index']['fanout']-1)/1e9,
            note='Disk/RAM estimate only. No latency or recall guarantee.'), indent=2))
    elif command == 'ingest':
        from .data import ingest
        result = ingest(args['config'], args['output']); print(dict(points=result['total_points'], series=len(result['series'])))
    elif command == 'train':
        from .train import fit
        print(fit(args['store'], args['config'], args['output'], args['stage'], args['device'], args['prior'], args['resume']))
    elif command == 'index':
        from .index import build
        result = build(args['store'], args['checkpoint'], args['output'], args['device'])
        print(dict(windows=result['windows'], payload_bytes=result['payload_bytes'], seconds=result['build_seconds']))
    elif command == 'evaluate':
        from .evaluate import evaluate
        evaluate(args['store'], args['checkpoint'], args['index'], args['output'], args['split'], args['device'],
                 args['leaf_budgets'], args['time_budget_ms'], args['oversample'])
    elif command == 'analyze':
        from .analyze import analyze
        analyze(args['results'], args['prior_training'], args['encoder_training'])
        print(Path(args['results'])/'analysis_bundle.zip')
    elif command == 'query':
        import numpy as np
        from .inference import Retriever
        from .common import write_json
        r = Retriever(args['store'], args['checkpoint'], args['index'], args['device'])
        try:
            result = r.retrieve(np.load(args['history'], allow_pickle=False), args['sid'], args['start'], leaf_budget=args['leaf_budget'])
            write_json(args['output'], dict(hits=result['hits'], stats=result['stats'], forecast=result['prediction'].tolist()))
        finally: r.close()
    elif command == 'compare':
        from .compare import compare
        compare(args['first'], args['second'], args['output']); print(args['output'])


if __name__ == '__main__':
    main()
