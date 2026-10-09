"""Recompute one held-out history-only query against a saved benchmark forecast."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import write_json
from .query_retrieval_finetune import predict


def verify(root, source, name, variant='rag', device='cpu'):
    folder=root/name
    records=json.loads((folder/variant/'predictions.json').read_text())
    r=records[0]  # Fixed first query, never chosen by forecast quality.
    with np.load(folder/'memory/test.npz',allow_pickle=False) as data:
        qi=r['query']; x=data['x'][qi].copy()
        assert (int(data['sid'][qi]),int(data['start'][qi]))==(r['sid'],r['start'])
    result=predict(source/name/'store',source/name/'train/best.pt',source/name/'index_a',
        folder/variant/'best.pt',x,r['sid'],r['start'],device)
    saved_hits=[json.loads(line) for line in (folder/'memory/test_retrieval.jsonl').read_text().splitlines()][qi]['hits']
    assert [(h['sid'],h['start']) for h in result['hits']]==[(h['sid'],h['start']) for h in saved_hits]
    expected=np.asarray(r['prediction']); actual=np.asarray(result['prediction'])
    np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-4)
    output=dict(dataset=name,variant=variant,fusion=result['fusion'],device=device,query=qi,
        candidate_identities_equal=True,max_abs_difference=float(np.abs(actual-expected).max()),
        rtol=2e-4,atol=2e-4)
    write_json(folder/variant/'query_verification.json',output)
    print(json.dumps(output))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--dataset',default='ETTh1')
    p.add_argument('--variant',default='rag')
    p.add_argument('--device',default='cpu')
    args=p.parse_args(); verify(Path(args.root),Path(args.source),args.dataset,args.variant,args.device)
