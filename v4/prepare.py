"""Create a train-only forecasting index from an existing raw time-series index."""
import argparse
import json
from pathlib import Path
import numpy as np
from forecast import series_key
from model import LearnedShapeModel
from index import IndexBuilder
from cascade import Cascade


def load_series(path):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    raw = np.memmap(path/'raw.bin', dtype='<f8', mode='r')
    groups = {}
    for s in manifest['series']:
        groups.setdefault(series_key(s), []).append(s)
    result = []
    for chunks in groups.values():
        chunks.sort(key=lambda s: s.get('start', 0))
        first = int(chunks[0].get('start', 0))
        end = max(int(s.get('start', 0))+s['n'] for s in chunks)
        values = np.full(end-first, np.nan)
        for s in chunks:
            a = int(s.get('start', 0))-first
            x = raw[s['offset']:s['offset']+s['n']]
            prev = values[a:a+len(x)]; known = np.isfinite(prev)
            if not np.array_equal(prev[known], x[known]):
                raise ValueError('Conflicting overlapping chunks')
            values[a:a+len(x)] = x
        if not np.isfinite(values).all():
            raise ValueError('Gaps/nonfinite data: split into contiguous runs before this experiment')
        meta = {k: v for k, v in chunks[0].items()
                if k not in ('offset', 'n', 'row_start', 'row_stride', 'row_offset')}
        meta['start'] = first
        result.append((meta, values))
    del raw
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-index', required=True)
    ap.add_argument('--output', default='results/memory')
    ap.add_argument('--train-fraction', type=float, default=.6)
    ap.add_argument('--length', type=int, default=244)
    args = ap.parse_args()
    if not 0 < args.train_fraction < 1:
        raise ValueError('Train fraction must be between zero and one')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    source = load_series(args.source_index)
    rng = np.random.default_rng(20260916)
    samples = []
    for meta, x in source:
        n = int(len(x)*args.train_fraction)
        if n >= args.length:
            starts = rng.integers(0, n-args.length+1, 64)
            samples.extend(x[p:p+args.length] for p in starts)
    model = LearnedShapeModel(length=args.length).fit(samples)
    builder = IndexBuilder(out/'base', model)
    for meta, x in source:
        n = int(len(x)*args.train_fraction)
        builder.add(x[:n], dict(meta, original_n=len(x), train_end=int(meta['start'])+n))
    builder.finish()
    Cascade.build(out/'base', out/'cascade')
    config = dict(source_index=str(Path(args.source_index).resolve()),
                  train_fraction=args.train_fraction, length=args.length,
                  original_unique_points=sum(len(x) for _, x in source),
                  train_points=sum(int(len(x)*args.train_fraction) for _, x in source),
                  logical_series=len(source), model_fit='training prefixes only')
    (out/'experiment.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(config, indent=2))


if __name__ == '__main__':
    main()
