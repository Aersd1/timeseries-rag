"""Reconstruct training-only fits, conservative policies and all raw metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import sha256,write_json
from .benchmark_error_correction import load
from .benchmark_residual_adapter import feature
from .residual_adapter import train
from .benefit_rerank import covariance
from .safe_error_policy import bound_delta,choose_policy,apply_policy
from .benchmark_state_recall import metric


def verify(root,source):
    root=Path(root);records=[]
    for path in sorted(root.glob('*/run.json')):
        folder=path.parent;run=json.loads(path.read_text(encoding='utf-8'));cache=Path(run['source_cache'])
        assert sha256(cache/'manifest.json')==run['source_manifest_sha256']
        for name,key in [('selection.json','selection_sha256'),('policy.json','policy_sha256'),('predictions.npz','predictions_sha256')]:
            assert sha256(folder/name)==run[key]
        selections=json.loads((folder/'selection.json').read_text(encoding='utf-8'))
        policies=json.loads((folder/'policy.json').read_text(encoding='utf-8'))
        data={sp:load(cache,sp) for sp in ['train','validation','test']};store=Store(Path(source)/folder.name/'store')
        deltas={kind:{sp:np.zeros_like(data[sp]['base']) for sp in ['validation','test']} for kind in policies}
        try:
            for s in store.series:
                sid=s['sid'];ids=np.flatnonzero(data['train']['sid']==sid)
                for kind in policies:
                    f=feature(data['train'],ids,s,run['period'],kind=='with_rag')
                    rebuilt,selection=train(f,data['train']['target'][ids],data['train']['start'][ids])
                    assert selection==selections[str(sid)][kind]
                    name=f'adapter_{sid}_{kind}.npz'
                    if 'adapter_hashes' in run:assert sha256(folder/name)==run['adapter_hashes'][name]
                    with np.load(folder/name,allow_pickle=False) as state:
                        for k in state.files:np.testing.assert_allclose(state[k],rebuilt[k],rtol=0,atol=1e-10)
                    for split in ['validation','test']:
                        current=data[split];rows=np.flatnonzero(current['sid']==sid)
                        prediction=covariance(rebuilt,feature(current,rows,s,run['period'],kind=='with_rag'))
                        deltas[kind][split][rows]=bound_delta(prediction*current['std'][rows,None],current['raw_quantiles'][rows])/current['std'][rows,None]
            val=data['validation'];test=data['test']
            with np.load(folder/'predictions.npz',allow_pickle=False) as f:
                for kind in policies:
                    policy=choose_policy(val['target'],{kind:deltas[kind]['validation']},val['sid'],val['start'])
                    assert policy==policies[kind]
                    forecast=test['base']+apply_policy(policy,{kind:deltas[kind]['test']},test['sid'])*test['std'][:,None]
                    np.testing.assert_array_equal(f['adapter_'+kind],forecast)
                for method,m in run['metrics'].items():assert metric(f[method],test)==m
            records.append(dict(dataset=folder.name,variables=len(store.series),train_only_models_and_folds_rebuilt=True,
                validation_policies_rebuilt=True,all_raw_metrics_recomputed=True))
            print('VERIFIED RESIDUAL',records[-1],flush=True)
        finally:store.close()
    write_json(root/'verification.json',dict(complete=True,records=records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    a=p.parse_args();verify(a.root,a.source)
