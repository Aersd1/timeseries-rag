"""Rebuild joint/group training and routing; audit chronology and new outcomes."""
import argparse
import json
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import sha256,write_json
from .benchmark_error_correction import load
from .benchmark_future_state_rag import experiment as joint_experiment
from .benchmark_group_future_state import experiment as group_experiment
from .benchmark_state_recall import metric,interval


def verify(root,source,rebuild_root,names):
    records=[]
    for name in names:
        folder=Path(root)/name;run=json.loads((folder/'run.json').read_text(encoding='utf-8'));checkpoint=Path(run['source_checkpoint'])
        base=json.loads((checkpoint/'run.json').read_text(encoding='utf-8'));cache=Path(base['source_cache']);store=Store(Path(source)/name/'store')
        try:
            pairs=0
            for split in ['train','validation','test']:
                data=load(cache,split)
                for s in store.series:
                    ids=np.flatnonzero(data['sid']==s['sid']);starts=data['candidate_starts'][ids]
                    assert np.all(starts+192<=data['start'][ids,None]);assert np.all(starts+192<=s['memory_end']);pairs+=starts.size
            for sid in np.unique(load(cache,'test')['sid']):
                assert store.series[int(sid)]['validation_end']>=store.series[int(sid)]['train_end']
        finally:store.close()
        rebuilt=Path(rebuild_root)/name
        if run.get('grouped'):group_experiment(checkpoint,Path(source)/name,rebuilt)
        else:joint_experiment(checkpoint,Path(run['source_complementary']),Path(source)/name,rebuilt)
        for path in folder.glob('*encoder*.npz'):
            with np.load(path,allow_pickle=False) as a,np.load(rebuilt/path.name,allow_pickle=False) as b:
                assert a.files==b.files
                for key in a.files:np.testing.assert_allclose(a[key],b[key],rtol=0,atol=1e-9)
        original_selection=json.loads((folder/'selection.json').read_text(encoding='utf-8'))
        regenerated_selection=json.loads((rebuilt/'selection.json').read_text(encoding='utf-8'))
        assert original_selection==regenerated_selection
        assert json.loads((folder/'policy.json').read_text(encoding='utf-8'))==json.loads((rebuilt/'policy.json').read_text(encoding='utf-8'))
        with np.load(folder/'predictions.npz',allow_pickle=False) as a,np.load(rebuilt/'predictions.npz',allow_pickle=False) as b:
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
            for method,m in run['metrics'].items():assert metric(a[method],a)==m
        records.append(dict(dataset=name,grouped=run.get('grouped',False),all_models_selections_and_policies_rebuilt=True,
            all_predictions_and_metrics_rebuilt=True,nonoverlapping_candidate_pairs=pairs))
        print('VERIFIED FUTURE ROUTER',records[-1],flush=True)
    write_json(Path(root)/'verification.json',dict(complete=True,records=records))


def verify_remaining(root,source):
    records=[]
    for path in sorted(Path(root).glob('*/run.json')):
        run=json.loads(path.read_text(encoding='utf-8'));folder=path.parent;contract=json.loads((folder/'contract.json').read_text(encoding='utf-8'))
        assert sha256(folder/'contract.json')==run['contract_sha256'];assert sha256(folder/'predictions.npz')==run['predictions_sha256']
        assert sha256(Path(run['source_checkpoint'])/'run.json')==contract['source_run_sha256']
        old=load(Path(contract['original_test_cache']),'test');prior=[old]
        for file,digest in contract['excluded_artifacts'].items():
            assert sha256(file)==digest
            with np.load(file,allow_pickle=False) as f:prior.append(dict(sid=f['sid'].copy(),start=f['start'].copy()))
        store=Store(Path(source)/folder.name/'store')
        try:
            with np.load(folder/'predictions.npz',allow_pickle=False) as f:
                for s in store.series:
                    ids=np.flatnonzero(f['sid']==s['sid']);starts=f['start'][ids]
                    np.testing.assert_array_equal(starts,contract['references'][str(s['sid'])]);assert np.all(np.diff(starts)>=192)
                    for earlier in prior:assert np.all(abs(starts[:,None]-earlier['start'][earlier['sid']==s['sid']][None,:])>=192)
                    assert np.all(f['selected_starts'][ids]+192<=starts[:,None]);assert np.all(f['selected_starts'][ids]+192<=s['memory_end'])
                    np.testing.assert_array_equal(f['raw_y'][ids],np.stack([store.window(s['sid'],p+96,96) for p in starts]))
                    np.testing.assert_array_equal(f['std'][ids],np.full(len(ids),max(s['memory_std'],1e-6),dtype=np.float32))
                for method,m in run['metrics'].items():assert metric(f[method],f)==m
                assert interval(f['moirai_direct'],f['routed'],f)==run['gain_interval']
                assert interval(f['encoder_only'],f['routed'],f)==run['rag_interval']
            records.append(dict(dataset=folder.name,queries=run['queries'],all_previous_and_internal_episode_overlap=0,
                frozen_before_outcomes=True,outcomes_and_metrics_verified=True))
            print('VERIFIED REMAINING FUTURE WINDOWS',records[-1],flush=True)
        finally:store.close()
    write_json(Path(root)/'verification.json',dict(complete=True,records=records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--rebuild-root');p.add_argument('--datasets',nargs='+');p.add_argument('--remaining',action='store_true');a=p.parse_args()
    if a.remaining:verify_remaining(a.root,a.source)
    else:verify(a.root,a.source,a.rebuild_root,a.datasets)
