"""Single-query analog forecasting; accepts only the known query past."""
import argparse
import json
import time
from pathlib import Path
from forecast import ForecastIndex, transfer, aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--memory', default='results/memory')
    ap.add_argument('--query-json', required=True)
    ap.add_argument('--horizon', type=int, default=96)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--capacity', type=int, default=8192)
    ap.add_argument('--scope', choices=('same_series', 'pooled_train'), default='same_series')
    ap.add_argument('--selection', choices=('ordinary', 'episodes'), default='episodes')
    ap.add_argument('--transfer', choices=('mean_std', 'endpoint', 'affine'), default='endpoint')
    ap.add_argument('--aggregation', choices=('mean', 'median'), default='mean')
    ap.add_argument('--output', default='results/prediction.json')
    args = ap.parse_args()
    payload = json.loads(Path(args.query_json).read_text(encoding='utf-8'))
    engine = ForecastIndex(Path(args.memory)/'cascade')
    try:
        began = time.perf_counter()
        result = engine.search(payload['values'], args.horizon, payload, int(payload['start']),
                               args.scope, args.capacity, args.k)
        hits = result[args.selection]
        if not hits:
            raise ValueError('No eligible historical neighbors; check scope, source identity and query start')
        past, future = engine.arrays(hits, args.horizon)
        mapped = transfer(payload['values'], past, future, args.transfer)
        prediction, effective, tau = aggregate(mapped, [hit[0] for hit in hits], args.aggregation)
        locations = []
        for d, sid, p in hits:
            hit = engine.idx.hit(int(sid), int(p), float(d), False)
            hit.update(future_start=hit['start']+engine.m, future_end=hit['start']+engine.m+args.horizon)
            locations.append(hit)
        output = dict(prediction=prediction.tolist(), hits=locations, horizon=args.horizon,
                      method=args.transfer, aggregation=args.aggregation, selection=args.selection,
                      requested_k=args.k, returned_k=len(hits), tau=tau, effective_neighbors=effective,
                      episode_complete=result['episode_complete'], retrieval_ms=result['retrieval_ms'],
                      forecast_ms=(time.perf_counter()-began)*1000,
                      scope=args.scope, self_exclusion='same logical series: candidate future end <= query start')
        dest = Path(args.output); dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps({k: output[k] for k in ('returned_k', 'retrieval_ms', 'forecast_ms')}, indent=2))
    finally:
        engine.close()


if __name__ == '__main__': main()
