"""Run frozen retrieval preparation, paired Moirai adaptation, and audited reporting."""
import argparse
import gc
from pathlib import Path
import torch
from .common import fresh_dir, write_json, sha256
from .prepare_retrieval_memory import prepare
from .train_retrieval_finetune import train
from .report_retrieval_finetune import report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--output',required=True)
    p.add_argument('--datasets',nargs='+',default=['ETTh1','ETTh2','ETTm1','ETTm2'])
    p.add_argument('--device',default='cuda')
    p.add_argument('--memory-device',default='cpu')
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=32)
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1:
        raise ValueError('Positive training budget required')
    root,source=fresh_dir(args.output),Path(args.source)
    torch.set_num_threads(4)
    write_json(root/'protocol.json',dict(datasets=args.datasets,variants=['plain','rag'],epochs=args.epochs,
        batch_size=args.batch_size,learning_rate=.0001,history_weight=.2,future_weight=.8,
        candidates=50,top_k=5,device=args.device,retriever='frozen A learned vectors',source=str(source),
        code_sha256={path:sha256(path) for path in ['v7/retrieval_finetune.py','v7/train_retrieval_finetune.py',
            'v7/prepare_retrieval_memory.py','v7/benchmark_retrieval_finetune.py']}))
    for name in args.datasets:
        prepare(source,root,name,args.memory_device)
        gc.collect()
        for variant in ['plain','rag']:
            train(root,source,name,variant,args.device,args.epochs,args.batch_size)
            gc.collect()
            if str(args.device).startswith('cuda'):
                torch.cuda.empty_cache()
    write_json(root/'complete.json',dict(complete=True,datasets=args.datasets,variants=['plain','rag']))
    report(root,source)


if __name__=='__main__':
    main()
