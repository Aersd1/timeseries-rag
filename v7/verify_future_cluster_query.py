"""Reload scratch encoders and reproduce a fixed, history-only benchmark query."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from .query_future_cluster import predict
from .common import write_json


def verify(root,source,name):
    folder=root/name
    expected=json.loads((folder/'predictions.json').read_text())[0]
    retrieval=[json.loads(line) for line in (folder/'retrieval.jsonl').read_text().splitlines()][0]
    with np.load(folder/'test.npz',allow_pickle=False) as data:
        x=data['x'][0].copy(); scale=max(float(x.std()),float(data['floor'][0]))
    checks=[]
    for mode in ['scratch_embedding','cluster_embedding','future_cluster','history_future_cluster','history_future_regression']:
        result=predict(folder,source/name/'store',x,expected['sid'],expected['start'],mode,'cpu')
        actual=np.asarray(result['prediction']); saved=np.asarray(expected['predictions'][mode])
        assert [r['start'] for r in result['hits']]==retrieval['methods'][mode]['starts']
        # FP32 device differences in very close dense neighbors can amplify score
        # weights. Compare in query-history units, not an arbitrary raw-unit epsilon.
        maximum=float(np.abs(actual-saved).max())
        assert maximum/scale<=1e-4, (mode,maximum,scale)
        checks.append(dict(mode=mode,max_abs_difference=maximum,max_history_scaled_difference=maximum/scale,
            normalized_tolerance=1e-4,same_candidate_identities=True))
    assert not any(m.startswith('uni2ts') or m=='v7.moirai' for m in sys.modules)
    record=dict(dataset=name,query=0,checks=checks,no_pretrained_forecaster_imported=True)
    write_json(folder/'query_verification.json',record); print(json.dumps(record,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',required=True)
    p.add_argument('--source',default='v7/runs/all_datasets_20261008'); p.add_argument('--dataset',default='ETTh1')
    args=p.parse_args(); verify(Path(args.root),Path(args.source),args.dataset)
