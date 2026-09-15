import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from pathlib import Path
import numpy as np
from index import SearchIndex

root=Path('results/benchmark_paa64');report=json.loads((root/'metrics.json').read_text(encoding='utf-8'))
idx=SearchIndex(report['index']);idx.warm_metadata();rows=[]
for e in [e for e in report['examples'] if e['metric']=='dtw']:
    truth=idx.search(e['query'],metric='dtw',k=10);cutoff=truth['hits'][-1]['squared_distance']
    runs=[idx.robust_search(e['query'],k=10) for _ in range(3)]
    for r in runs:
        assert r['certified']
        assert all(h['squared_distance']<=cutoff+2e-5 for h in r['hits'])
    row=dict(group=e['group'],p50_ms=float(np.median([r['total_ms'] for r in runs])),
        max_ms=max(r['total_ms'] for r in runs),recall10=1.,certified=True,
        seed_stage_ms=runs[-1]['seed_stage_ms'],dtw_positions=runs[-1]['positions_verified'],
        ed_positions=runs[-1]['seed_positions_verified'])
    rows.append(row);print(row,flush=True)
(root/'robust.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');idx.close()
