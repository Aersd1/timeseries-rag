"""Independently audit remaining test episodes and recompute forecast errors."""
import argparse
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from v6.data import Windows
from .common import sha256,write_json


def verify(root,source):
    summary=json.loads((root/'extra_test_summary.json').read_text()); checks=[]
    assert summary['complete'] and summary['used_for_selection'] is False
    for result in summary['results']:
        name=result['dataset']; folder=root/name
        run=json.loads((folder/'test/run.json').read_text()); c=run['config']; span=c['length']+max(c['horizons'])
        assert result['gate_sha256']==sha256(folder/'quality_gate.pt') and result['policy_sha256']==sha256(root/'policy.json')
        original=Windows(source/name/'store',c,'test'); store=Store(source/name/'store')
        try:
            records=json.loads((folder/'extra_test/predictions.json').read_text()); assert len(records)==result['queries']
            totals={m:[] for m in ['a50','a100','moirai_direct','a100_adaptive']}
            ids={(r['sid'],r['start']) for r in records}; assert len(ids)==len(records)
            for sid,start in ids:
                assert all(qsid!=sid or abs(start-qstart)>=span for qsid,qstart in original.refs)
                assert all(other_sid!=sid or other_start==start or abs(start-other_start)>=span for other_sid,other_start in ids)
            for r in records:
                s=store.series[r['sid']]; start=r['start']
                assert s['validation_end']<=start and start+span<=s['n']
                for hit in r['selected']:
                    assert hit['sid']==r['sid'] and hit['start']+span<=min(start,s['memory_end'])
                y=store.window(r['sid'],start+c['length'],max(c['horizons'])).astype(float)
                for method,values in r['predictions'].items():
                    error=np.asarray(values,dtype=float)-y
                    score=float(sum(np.dot(error[:h],error[:h])/h for h in c['horizons'])/
                        len(c['horizons'])/max(s['memory_std'],1e-6)**2)
                    np.testing.assert_allclose(score,r['scores'][method],rtol=1e-10,atol=1e-10)
                    totals[method].append(score)
            for method,values in totals.items(): np.testing.assert_allclose(np.mean(values),result[method],rtol=1e-10,atol=1e-10)
            checks.append(dict(dataset=name,queries=len(records),recomputed_query_method_scores=len(records)*len(totals),
                disjoint_from_previous_queries=True,selected_candidate_boundary_violations=0))
        finally: original.store.close(); store.close()
    report=dict(complete=True,checks=checks); write_json(root/'extra_test_verification.json',report); print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',required=True)
    p.add_argument('--source',default='v7/runs/all_datasets_20261008'); args=p.parse_args()
    verify(Path(args.root),Path(args.source))
