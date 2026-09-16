"""Run from repository root: python -m v5 --help."""
import argparse
import importlib.util
import glob
from pathlib import Path
from .common import read_json, validate_config, environment


def main():
    parser = argparse.ArgumentParser(description='V5 unified temporal tokens; training runs only when explicitly invoked')
    sub = parser.add_subparsers(dest='command', required=True)
    doctor = sub.add_parser('doctor'); doctor.add_argument('--config', required=True)
    ingest = sub.add_parser('ingest'); ingest.add_argument('--config', required=True); ingest.add_argument('--output', required=True)
    teacher = sub.add_parser('teacher'); teacher.add_argument('--store', required=True); teacher.add_argument('--output', required=True)
    train = sub.add_parser('train'); train.add_argument('--store', required=True); train.add_argument('--teacher', required=True)
    train.add_argument('--output', required=True); train.add_argument('--device', default='cpu'); train.add_argument('--resume', action='store_true'); train.add_argument('--config')
    index = sub.add_parser('index'); index.add_argument('--store', required=True); index.add_argument('--checkpoint', required=True)
    index.add_argument('--output', required=True); index.add_argument('--device', default='cpu')
    evaluate = sub.add_parser('evaluate'); evaluate.add_argument('--store', required=True); evaluate.add_argument('--teacher', required=True)
    evaluate.add_argument('--checkpoint', required=True); evaluate.add_argument('--index', required=True); evaluate.add_argument('--output', required=True)
    evaluate.add_argument('--split', choices=('validation','test'), default='test'); evaluate.add_argument('--device', default='cpu')
    analyze = sub.add_parser('analyze'); analyze.add_argument('--results', required=True); analyze.add_argument('--training')
    args = parser.parse_args()
    if args.command == 'doctor':
        c = validate_config(read_json(args.config)); print(environment())
        for source in c['data']['sources']:
            files = glob.glob(source['glob'],recursive=True)
            print(f"CSV matches: {len(files)} for group {source.get('group','custom')}")
            if not files: raise SystemExit('Edit config CSV paths/columns before running')
        if importlib.util.find_spec('torch') is None: raise SystemExit('PyTorch missing; follow SERVER_GUIDE.md')
        import torch
        print(dict(torch=torch.__version__,cuda=torch.cuda.is_available(),gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None))
        print('No training or data import performed. Compile V4 native kernel before teacher stage.')
    elif args.command == 'ingest':
        from .data import ingest
        result = ingest(args.config,args.output); print(dict(points=result['total_points'],series=len(result['series'])))
    elif args.command == 'teacher':
        from .teacher import generate
        result = generate(args.store,args.output); print(result['counts'])
    elif args.command == 'train':
        from .train import fit
        print(fit(args.store,args.teacher,args.output,args.device,args.resume,args.config))
    elif args.command == 'index':
        from .token_index import build
        result = build(args.store,args.checkpoint,args.output,args.device); print(result['compression'])
    elif args.command == 'evaluate':
        from .evaluate import evaluate
        evaluate(args.store,args.teacher,args.checkpoint,args.index,args.output,args.split,args.device)
    elif args.command == 'analyze':
        from .analyze import analyze
        analyze(args.results,args.training); print(Path(args.results)/'analysis_bundle.zip')


if __name__ == '__main__': main()
