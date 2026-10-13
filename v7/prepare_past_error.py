"""Frozen Moirai64-to32 probe; every probe target is inside96-point history."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .common import sha256,write_json
from .benchmark_error_correction import load
from .moirai import MoiraiAdapter,MODEL_ID,MODEL_REVISION


def prepare(cache,out,device):
    meta=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    if meta['model_identity']['model_id']!=MODEL_ID or meta['model_identity']['revision']!=MODEL_REVISION:raise ValueError('Wrong model')
    contract=dict(source_manifest_sha256=sha256(cache/'manifest.json'),model_identity=meta['model_identity'],context=64,horizon=32)
    if (out/'manifest.json').exists():
        saved=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if saved['contract']!=contract:raise ValueError('Changed probe contract')
        for split,digest in saved['files'].items():
            if sha256(out/f'{split}.npz')!=digest:raise ValueError('Changed probe cache')
        return
    out.mkdir(parents=True,exist_ok=True);adapter=MoiraiAdapter.pretrained(meta['moirai'],device);files={}
    for split in ['train','validation','test']:
        data=load(cache,split);histories=data['raw_x'];quantiles=[]
        for offset in range(0,len(histories),128):
            quantiles.extend(adapter.forecast(histories[offset:offset+128,:64],32).cpu().numpy())
        q=np.stack(quantiles);error=histories[:,64:]-q[:,4]
        np.savez_compressed(out/f'{split}.npz',quantiles=q,error=error,sid=data['sid'],start=data['start'])
        files[split]=sha256(out/f'{split}.npz');print(cache.name,'observed probe',split,len(error),flush=True)
    write_json(out/'manifest.json',dict(complete=True,contract=contract,files=files,input='Only first64 history points; compared with observed final32 history points'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.set_num_threads(4)
    for name in a.datasets:prepare(Path(a.cache)/name,Path(a.output)/name,a.device)
