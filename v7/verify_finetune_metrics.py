"""Independently recompute saved forecast metrics from targets and predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import sha256, write_json


def verify(root,source):
    complete=json.loads((root/'complete.json').read_text())
    if not complete['complete']:
        raise ValueError('Experiment not complete')
    checks=[]
    for name in complete['datasets']:
        folder=root/name
        meta=json.loads((folder/'memory/manifest.json').read_text())
        catalog=json.loads((source/name/'store/catalog.json').read_text())
        with np.load(folder/'memory/test.npz',allow_pickle=False) as archive:
            data={k:archive[k].copy() for k in archive.files}
        for variant in complete['variants']:
            forecasts=json.loads((folder/variant/'predictions.json').read_text())
            assert len(forecasts)==len(data['x'])
            scores={}
            for line in (folder/variant/'metrics.jsonl').read_text().splitlines():
                row=json.loads(line); key=(row['query'],row['method'],row['horizon'])
                assert key not in scores
                scores[key]=row
            checked=0
            for qi,r in enumerate(forecasts):
                assert r['query']==qi and (r['sid'],r['start'])==(int(data['sid'][qi]),int(data['start'][qi]))
                values={'finetuned':r['prediction'], 'frozen':data['frozen_prediction'][qi],
                    'analog':data['analog_prediction'][qi]}
                if variant!='plain':
                    values['memory_disabled']=r['memory_disabled']
                target=data['y'][qi].astype(np.float64)
                variance=max(catalog['series'][r['sid']]['memory_std'],1e-6)**2
                for method,prediction in values.items():
                    prediction=np.asarray(prediction,dtype=np.float64)
                    assert prediction.shape==target.shape and np.isfinite(prediction).all()
                    for horizon in meta['config']['horizons']:
                        error=prediction[:horizon]-target[:horizon]
                        mse=float(np.dot(error,error)/horizon)
                        mae=float(sum(abs(float(e)) for e in error)/horizon)
                        row=scores[(qi,method,horizon)]
                        assert (row['sid'],row['start'])==(r['sid'],r['start'])
                        np.testing.assert_allclose([row['mse'],row['mae'],row['nmse']],
                            [mse,mae,mse/variance],rtol=1e-10,atol=1e-10)
                        checked+=1
            assert checked==len(scores)
            checks.append(dict(dataset=name,variant=variant,metric_rows=checked,all_scores_recomputed=True))
    result=dict(complete=True,checks=checks,code_sha256=sha256(__file__))
    write_json(root/'metric_verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    args=p.parse_args(); verify(Path(args.root),Path(args.source))
