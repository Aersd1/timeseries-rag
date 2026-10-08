"""Cache history-only A50 reranking with fixed 0.8 future / 0.2 history scores."""
import argparse
import json
from pathlib import Path
import numpy as np
from v6.data import Windows, scale_floor
from .common import write_json, fresh_dir, sha256
from .learned_rerank import LearnedMoiraiReranker


def prepare(source, root, name, device='cpu'):
    src = source/name
    out = fresh_dir(root/name/'memory')
    engine = LearnedMoiraiReranker(src/'store', src/'train/best.pt', src/'index_a', device)
    c = engine.c
    run = dict(complete=False, dataset=name, config=c, retriever='a50_ordinary', candidates=50, top_k=5,
        future_weight=.8, history_weight=.2, scorer='frozen pretrained Moirai 2',
        data_id=engine.store.meta['data_id'], checkpoint_sha256=sha256(src/'train/best.pt'),
        index_sha256=sha256(src/'index_a/manifest.json'), splits={})
    write_json(out/'manifest.json', run)
    try:
        for split in ['train', 'validation', 'test']:
            ds = Windows(src/'store', c, split)
            arrays = {k:[] for k in ['x', 'y', 'floor', 'sid', 'start', 'examples', 'weights', 'valid', 'frozen_prediction', 'analog_prediction']}
            records, full = [], 0
            try:
                for qi in range(len(ds)):
                    item = ds[qi]
                    sid, start = int(item['sid']), int(item['start'])
                    result = engine.retrieve(item['x'], sid, start, history_weight=.2)
                    examples = np.zeros((5, c['length']+max(c['horizons'])), dtype=np.float32)
                    weights = np.zeros(5, dtype=np.float32)
                    valid = np.zeros(5, dtype=bool)
                    for i, hit in enumerate(result['hits']):
                        assert hit['sid']==sid and hit['start']+examples.shape[-1] <= min(start, engine.store.series[sid]['memory_end'])
                        past = engine.store.window(sid, hit['start'], c['length']).astype(float)
                        both = engine.store.window(sid, hit['start'], examples.shape[-1]).astype(float)
                        scale = max(float(past.std()), scale_floor(engine.store.series[sid], c))
                        examples[i] = (both-past.mean())/scale
                        weights[i] = hit['weight']
                        valid[i] = True
                    for key in ['x', 'y', 'floor', 'sid', 'start']:
                        arrays[key].append(item[key])
                    for key, value in [('examples', examples), ('weights', weights), ('valid', valid),
                        ('frozen_prediction', result['direct_prediction']), ('analog_prediction', result['prediction'])]:
                        arrays[key].append(value)
                    records.append(dict(query=qi, sid=sid, start=start, hits=result['hits'],
                        returned_candidates=result['stats']['returned_candidates']))
                    full += len(result['hits'])==5
                    if (qi+1)%32==0 or qi+1==len(ds):
                        print(name, split, qi+1, '/', len(ds), flush=True)
                np.savez_compressed(out/(split+'.npz'), **{k:np.stack(v) for k,v in arrays.items()})
                with (out/(split+'_retrieval.jsonl')).open('w', encoding='utf-8') as f:
                    for record in records:
                        f.write(json.dumps(record, allow_nan=False)+'\n')
                run['splits'][split] = dict(queries=len(ds), full_top5=full, sha256=sha256(out/(split+'.npz')))
                write_json(out/'manifest.json', run)
            finally:
                ds.store.close()
        run['complete'] = True
        write_json(out/'manifest.json', run)
    finally:
        engine.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='v7/runs/all_datasets_20261008')
    p.add_argument('--output', default='v7/runs/moirai_rag_finetune_20261008')
    p.add_argument('--datasets', nargs='+', default=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'])
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    for name in args.datasets:
        prepare(Path(args.source), Path(args.output), name, args.device)
