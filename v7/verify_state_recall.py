"""Audit selection, history-only candidate inputs, chronology and saved metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import sha256,write_json
from .state_recall import STATE_WEIGHTS
from .benefit_rerank import RIDGES,guarded_policy,apply_policy
from .benchmark_state_recall import Builder,corrections,metric


def verify(root,source):
    root=Path(root);audits=[]
    for path in sorted(root.glob('*/run.json')):
        folder=path.parent;run=json.loads(path.read_text(encoding='utf-8'));cache=Path(run['source_cache'])
        assert sha256(cache/'manifest.json')==run['source_manifest_sha256']
        assert sha256(folder/'selection.json')==run['selection_sha256']
        selection=json.loads((folder/'selection.json').read_text(encoding='utf-8'))
        for w,s in selection['candidates'].items():
            for fold in s['folds']:assert fold['train_max_episode_end']<=fold['heldout_first_start']
            assert s['selected_penalty']==min(RIDGES,key=lambda p:(np.mean(s['forward_nmse'][str(p)]),-p))
        chosen=min(STATE_WEIGHTS,key=lambda w:(np.mean(selection['candidates'][str(w)]['forward_nmse'][str(selection['candidates'][str(w)]['selected_penalty'])]),w))
        assert chosen==run['selected_state_weight']
        builder=Builder(cache,Path(source)/folder.name);pairs=0
        try:
            for weight in sorted(set([0.,chosen])):
                label='utility_shape' if weight==0 else 'utility_state';saved=run['methods'][label]
                branch=folder/f'weight_{weight}'
                for file,key in [('ranker.npz','ranker_sha256'),('policy.json','policy_sha256'),
                    ('test.npz','test_sha256'),('predictions.npz','predictions_sha256')]:
                    assert sha256(branch/file)==saved[key]
                with np.load(branch/'ranker.npz',allow_pickle=False) as f:state={k:f[k].copy() for k in f.files}
                policy=json.loads((branch/'policy.json').read_text(encoding='utf-8'))['utility']
                val=builder.table('validation',weight);vr=corrections(state,val)
                assert guarded_policy(val['target'],vr['correction'],val['sid'],val['start'])==policy
                del val,vr
                test=builder.table('test',weight)
                with np.load(branch/'test.npz',allow_pickle=False) as f:
                    for key in f.files:np.testing.assert_array_equal(test[key],f[key])
                assert np.all(test['starts']+192<=test['start'][:,None])
                pairs+=test['starts'].size
                tr=corrections(state,test);forecast=test['base']+apply_policy(policy,tr['correction'],test['sid'])*test['std'][:,None]
                with np.load(branch/'predictions.npz',allow_pickle=False) as f:
                    for key in ['selected','weights','permitted']:np.testing.assert_array_equal(f[key],tr[key])
                    np.testing.assert_array_equal(f['prediction'],forecast)
                assert metric(forecast,test)==saved['metrics']
            assert sha256(folder/'predictions.npz')==run['predictions_sha256']
            with np.load(folder/'predictions.npz',allow_pickle=False) as f:
                table={k:f[k] for k in ['raw_y','std']}
                for method,saved in run['methods'].items():assert metric(f[method],table)==saved['metrics']
            audits.append(dict(dataset=folder.name,test_candidate_pairs_reconstructed=pairs,overlap_violations=0,
                train_only_hyperparameter_choice_checked=True,validation_policy_rebuilt=True,metrics_recomputed=True))
            print('VERIFIED STATE',audits[-1],flush=True)
        finally:builder.close()
    write_json(root/'verification.json',dict(complete=True,audits=audits))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    a=p.parse_args();verify(a.root,a.source)
