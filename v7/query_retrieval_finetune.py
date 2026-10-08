"""Predict from history using a selected fine-tuned Moirai checkpoint and A50 memory."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from v6.data import scale_floor
from .common import sha256, write_json
from .moirai import MoiraiAdapter
from .learned_rerank import LearnedMoiraiReranker
from .retrieval_finetune import RetrievalFineTune


def predict(store, checkpoint_a, index_a, checkpoint, x, sid, start, device='cpu'):
    checkpoint = Path(checkpoint)
    run = json.loads((checkpoint.parent/'run.json').read_text())
    if not run['complete'] or run['checkpoint_sha256'] != sha256(checkpoint):
        raise ValueError('Incomplete or changed fine-tuned artifact')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state.get('variant') != 'moirai_retrieval_finetune':
        raise ValueError('Expected fine-tuned Moirai artifact')
    c = state['config']
    engine = LearnedMoiraiReranker(store, checkpoint_a, index_a, device)
    try:
        if run['data_id'] != engine.store.meta['data_id'] or c != engine.c:
            raise ValueError('Fine-tuned model/retriever/store mismatch')
        result = engine.retrieve(x, sid, start, history_weight=.2)
        m, h = c['length'], max(c['horizons'])
        examples, weights, valid = np.zeros((5,m+h),np.float32), np.zeros(5,np.float32), np.zeros(5,bool)
        for i, hit in enumerate(result['hits']):
            past = engine.store.window(sid, hit['start'], m).astype(float)
            both = engine.store.window(sid, hit['start'], m+h).astype(float)
            examples[i] = (both-past.mean())/max(float(past.std()),scale_floor(engine.store.series[sid], c))
            weights[i], valid[i] = hit['weight'], True
        adapter = MoiraiAdapter.pretrained(c['moirai'], device)
        if adapter.architecture != state['architecture'] or adapter.identity != state['identity']:
            raise ValueError('Pretrained model identity changed')
        model = RetrievalFineTune(adapter,m,h,state['use_retrieval'],state['rank']).to(device)
        model.load_adaptation(state['adaptation']); model.eval()
        with torch.no_grad():
            inputs = [torch.as_tensor(v[None],device=device) for v in [np.asarray(x,np.float32),examples,weights,valid]]
            quantiles = model(*inputs).cpu().numpy()[0]
        levels = list(model.module.quantile_levels)
        return dict(prediction=quantiles[levels.index(.5)].tolist(), quantiles=quantiles.tolist(),
            quantile_levels=levels, hits=result['hits'], history_weight=.2, future_weight=.8,
            best_epoch=state['epoch'], use_retrieval=state['use_retrieval'],
            frozen_prediction=result['direct_prediction'].tolist(), analog_prediction=result['prediction'].tolist())
    finally:
        engine.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['store','checkpoint-a','index-a','checkpoint','history','output']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--sid',type=int,required=True)
    p.add_argument('--start',type=int,required=True)
    p.add_argument('--device',default='cpu')
    args = vars(p.parse_args())
    output, history = Path(args.pop('output')), args.pop('history')
    if output.exists():
        raise FileExistsError('Choose a fresh output')
    write_json(output,predict(x=np.load(history,allow_pickle=False),**args))
