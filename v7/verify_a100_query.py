"""Reproduce a fixed benchmark query with a reloaded reliability checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import write_json
from .query_a100_quality import predict


def verify(root,source,name):
    folder=root/name; row=json.loads((folder/'test/predictions.json').read_text())[0]
    with np.load(folder/'test/cache.npz',allow_pickle=False) as data: x=data['x'][0].copy()
    result=predict(folder/'quality_gate.pt',source/name/'store',source/name/'train/best.pt',source/name/'index_a',
        x,row['sid'],row['start'],'cpu')
    expected=np.asarray(row['predictions']['a100_adaptive']); actual=np.asarray(result['prediction'])
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-5)
    details=json.loads((folder/'test/retrieval.json').read_text())[0]['selected']['a100_diverse']
    assert [h['start'] for h in result['hits']]==[h['start'] for h in details]
    np.testing.assert_allclose(result['analogue_alpha'],row['deployed_alpha'],rtol=1e-6,atol=1e-6)
    record=dict(dataset=name,query=0,adopted=result['adopted'],candidate_ids_equal=True,
        max_abs_difference=float(np.abs(actual-expected).max()),analogue_alpha=result['analogue_alpha'])
    write_json(folder/'query_verification.json',record); print(json.dumps(record))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',required=True)
    p.add_argument('--source',default='v7/runs/all_datasets_20261008'); p.add_argument('--dataset',default='ETTm1')
    args=p.parse_args(); verify(Path(args.root),Path(args.source),args.dataset)
