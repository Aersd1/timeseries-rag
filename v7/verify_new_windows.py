"""Audit disjoint evaluation references, true outcomes and all reported metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import sha256,write_json
from .benchmark_error_correction import load
from .benchmark_state_recall import metric,interval


def verify(root,source):
    records=[]
    for path in sorted(Path(root).glob('*/run.json')):
        run=json.loads(path.read_text(encoding='utf-8'));folder=path.parent;checkpoint=Path(run['source_checkpoint'])
        contract=json.loads((folder/'evaluation_contract.json').read_text(encoding='utf-8'))
        assert sha256(folder/'evaluation_contract.json')==run['contract_sha256']
        assert sha256(folder/'predictions.npz')==run['predictions_sha256']
        assert sha256(checkpoint/'run.json')==contract['checkpoint_run_sha256']
        original=json.loads((checkpoint/'run.json').read_text(encoding='utf-8'));old=load(Path(original['source_cache']),'test')
        store=Store(Path(source)/folder.name/'store')
        try:
            with np.load(folder/'predictions.npz',allow_pickle=False) as f:
                for s in store.series:
                    sid=s['sid'];ids=np.flatnonzero(f['sid']==sid);starts=f['start'][ids]
                    np.testing.assert_array_equal(starts,contract['references'][str(sid)])
                    assert np.all(np.diff(starts)>=192)
                    assert np.all(abs(starts[:,None]-old['start'][old['sid']==sid][None,:])>=192)
                    assert np.all(starts>=s['validation_end']) and np.all(starts+192<=s['n'])
                    np.testing.assert_array_equal(f['raw_y'][ids],np.stack([store.window(sid,p+96,96) for p in starts]))
                    np.testing.assert_array_equal(f['std'][ids],np.full(len(ids),max(s['memory_std'],1e-6),dtype=np.float32))
                for name,m in run['metrics'].items():assert metric(f[name],f)==m
                assert interval(f['moirai_direct'],f['past_with_rag'],f)==run['gain_interval']
            records.append(dict(dataset=folder.name,queries=run['queries'],variables=run['variables'],outcomes_and_metrics_rebuilt=True,
                original_test_overlap=0,pairwise_new_episode_overlap=0,frozen_checkpoint_contract_verified=True))
            print('VERIFIED NEW WINDOWS',records[-1],flush=True)
        finally:store.close()
    write_json(Path(root)/'verification.json',dict(complete=True,records=records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    a=p.parse_args();verify(a.root,a.source)
