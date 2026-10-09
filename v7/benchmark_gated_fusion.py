"""Matched fusion ablations using immutable, previously audited A50 memory.

Reuse frozen/LoRA/legacy controls, then train gate-only, paired-without-dropout,
and paired-with-dropout models. Only validation selects adapter checkpoints.
"""
import argparse
import gc
import json
import shutil
from pathlib import Path
import torch
from .common import fresh_dir, sha256, write_json
from .train_retrieval_finetune import train
from .report_gated_fusion import report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous', default='v7/runs/moirai_rag_finetune_20261008')
    p.add_argument('--source', default='v7/runs/all_datasets_20261008')
    p.add_argument('--output', required=True)
    p.add_argument('--datasets', nargs='+', default=['ETTh1','ETTh2','ETTm1','ETTm2'])
    p.add_argument('--device', default='cuda')
    args=p.parse_args()
    previous, source, root=Path(args.previous), Path(args.source), fresh_dir(args.output)
    previous_protocol=json.loads((previous/'protocol.json').read_text())
    if not json.loads((previous/'complete.json').read_text())['complete']:
        raise ValueError('Previous controls must be complete')
    epochs, batch=previous_protocol['epochs'], previous_protocol['batch_size']
    modes=[('rag_gated','gated',0.,0.),('rag_paired','paired',0.,0.),('rag','paired',.2,.1)]
    variants=['plain','rag_legacy']+[v[0] for v in modes]
    torch.set_num_threads(4)
    reused={}
    for name in args.datasets:
        if name not in previous_protocol['datasets']:
            raise ValueError('Dataset absent from matched controls')
        folder=root/name; folder.mkdir()
        for src,dst in [('memory','memory'),('plain','plain'),('rag','rag_legacy')]:
            shutil.copytree(previous/name/src,folder/dst)
        reused[name]={}
        for variant in ['plain','rag_legacy']:
            run=json.loads((folder/variant/'run.json').read_text())
            if not run['complete'] or run.get('fusion','legacy')!='legacy':
                raise ValueError('Expected legacy/LoRA controls')
            if run['epochs_limit']!=epochs or run['batch_size']!=batch or run['learning_rate']!=.0001:
                raise ValueError('Control training budget differs')
            if run['checkpoint_sha256']!=sha256(folder/variant/'best.pt'):
                raise ValueError('Control checkpoint changed')
            reused[name][variant]={f:sha256(folder/variant/f) for f in ['best.pt','run.json','metrics.jsonl','predictions.json']}
    protocol=dict(datasets=args.datasets, variants=variants, epochs=epochs, batch_size=batch,
        history_weight=.2,future_weight=.8,candidates=50,top_k=5,device=args.device,
        memory_reuse=str(previous), reused_controls=reused,
        experiments=[dict(variant=v,fusion=f,candidate_dropout=c,memory_dropout=m) for v,f,c,m in modes],
        code_sha256={f:sha256(f) for f in ['v7/retrieval_finetune.py','v7/train_retrieval_finetune.py',
            'v7/benchmark_gated_fusion.py','v7/report_gated_fusion.py','v7/report_retrieval_finetune.py']})
    write_json(root/'protocol.json',protocol)
    for name in args.datasets:
        for variant,fusion,candidate_dropout,memory_dropout in modes:
            train(root,source,name,variant,args.device,epochs,batch,fusion,candidate_dropout,memory_dropout)
            gc.collect()
            if str(args.device).startswith('cuda'):
                torch.cuda.empty_cache()
    write_json(root/'complete.json',dict(complete=True,datasets=args.datasets,variants=variants))
    report(root,source)


if __name__=='__main__':
    main()
