"""Reconstruct ablations, boundaries, physical sources and actual inference."""
import argparse
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from .common import sha256,write_json
from .prepare_unified_residual import read
from .unified_residual import MODES,features,matched_starts,state_features
from .benchmark_unified_residual import trace,confirmed,apply_confirmed,predict,fixed_transfer
from .benchmark_state_recall import metric,interval
from .query_unified_residual import UnifiedResidualPredictor
from .unified_residual import UnifiedResidualFusion


def verify_derangement(folder,device='cpu'):
    from .benchmark_unified_derangement import derange
    run=read(folder/'derangement_run.json');contract=read(folder/'derangement_contract.json')
    assert sha256(folder/'derangement_contract.json')==run['contract_sha256'];assert sha256(folder/'derangement_predictions.npz')==run['predictions_sha256']
    assert sha256(folder/'run.json')==contract['source_run_sha256'];assert sha256(folder/'derangement_policy.json')==contract['policy_sha256'];assert sha256(folder/'derangement_validation.npz')==contract['validation_sha256']
    val=derange(trace(folder,'validation'));test=derange(trace(folder,'test'));train=derange(trace(folder,'train'))
    for data in [train,val,test]:
        assert np.all(data['shuffle_order']!=np.arange(10)[None])
        for j in [0,5]:np.testing.assert_array_equal(np.sort(data['shuffle_order'][:,j:j+5],1),np.tile(np.arange(j,j+5),(len(data['sid']),1)))
    del train
    meta=read(folder/'preparation.json');stats=read(folder/'stats.json');original=read(folder/'selection.json');policies=read(folder/'derangement_policy.json');models=[]
    assert contract['settings']==original['settings'];assert len(contract['models'])==3
    for r in contract['models']:
        path=folder/r['file'];assert sha256(path)==r['sha256'];assert len(r['losses'])==16;assert r['parameters']==original['models'][0]['parameters']
        saved=torch.load(path,map_location='cpu',weights_only=True);m=UnifiedResidualFusion(saved['variables'],saved['width']);m.load_state_dict(saved['state_dict']);models.append(m.to(device).eval())
    with np.load(folder/'derangement_validation.npz') as vf,np.load(folder/'derangement_predictions.npz') as sf,np.load(folder/'validation_predictions.npz') as ov,np.load(folder/'predictions.npz') as old:
        np.testing.assert_array_equal(vf['order'],val['shuffle_order']);np.testing.assert_array_equal(sf['order'],test['shuffle_order'])
        vp=np.mean([predict(m,val,stats,meta['period'],'shuffled',device) for m in models],0).astype(np.float32);np.testing.assert_allclose(vp,vf['raw'],rtol=2e-5,atol=2e-4)
        _,own=confirmed(val,vf['raw'],stats,val['raw_quantiles'][:,4]);_,extra=confirmed(val,vf['raw'],stats,ov['query_confirmed']);assert dict(own=own,extra=extra)==policies
        raw=np.mean([sf[f'seed_{s}'] for s in original['settings']['seeds']],0).astype(np.float32);np.testing.assert_array_equal(raw,sf['raw'])
        np.testing.assert_array_equal(apply_confirmed(test,raw,test['raw_quantiles'][:,4],own),sf['own']);np.testing.assert_array_equal(apply_confirmed(test,raw,old['query_confirmed'],extra),sf['extra'])
        for key,v in run['metrics'].items():assert metric(sf[key],test)==v
        ids=np.unique([0,len(test['sid'])//2,len(test['sid'])-1]);small={k:v[ids] for k,v in test.items()}
        for m,seed in zip(models,original['settings']['seeds']):np.testing.assert_allclose(predict(m,small,stats,meta['period'],'shuffled',device),sf[f'seed_{seed}'][ids],rtol=2e-5,atol=2e-4)
    r=dict(dataset=folder.name,complete=True,all_pairs_mismatched=True,within_route_value_set_preserved=True,same_parameters_and_training_budget=True,policies_predictions_rebuilt=True,device=device)
    print('VERIFIED DERANGEMENT',r,flush=True);return r


def verify_capacity(folder,device='cpu'):
    run=read(folder/'capacity_run.json');contract=read(folder/'capacity_contract.json')
    for file,key in [('capacity_contract.json','contract_sha256'),('capacity_validation.npz','validation_sha256'),('capacity_predictions.npz','prediction_sha256')]:assert sha256(folder/file)==run[key]
    assert sha256(folder/'run.json')==contract['source_run_sha256'];assert sha256(folder/'capacity_policy.json')==contract['policy_sha256']
    meta=read(folder/'preparation.json');stats=read(folder/'stats.json');original=read(folder/'selection.json')
    assert contract['settings']==original['settings'];assert len(contract['models'])==3
    assert contract['seeds']==[s+100 for s in original['settings']['seeds']]
    val=trace(folder,'validation');test=trace(folder,'test');models=[]
    for r in contract['models']:
        path=folder/r['file'];assert sha256(path)==r['sha256'];assert len(r['losses'])==16
        assert r['parameters']==original['models'][0]['parameters']
        saved=torch.load(path,map_location='cpu',weights_only=True);model=UnifiedResidualFusion(saved['variables'],saved['width']);model.load_state_dict(saved['state_dict']);models.append(model.to(device).eval())
    with np.load(folder/'capacity_validation.npz',allow_pickle=False) as vf,np.load(folder/'capacity_predictions.npz',allow_pickle=False) as sf,np.load(folder/'predictions.npz',allow_pickle=False) as old,np.load(folder/'validation_predictions.npz',allow_pickle=False) as ov:
        vp=np.mean([predict(m,val,stats,meta['period'],'query_only',device) for m in models],0).astype(np.float32)
        np.testing.assert_allclose(vp,vf['raw'],rtol=2e-5,atol=2e-4)
        _,policy=confirmed(val,vf['raw'],stats,ov['query_confirmed']);assert policy==read(folder/'capacity_policy.json')
        raw=np.mean([sf[f'seed_{s}'] for s in contract['seeds']],0).astype(np.float32);np.testing.assert_array_equal(raw,sf['raw_replica'])
        final=apply_confirmed(test,raw,old['query_confirmed'],policy);np.testing.assert_array_equal(final,sf['query_query']);assert metric(final,test)==run['metrics']
        assert metric(raw,test)==run['raw_replica_metrics']
        np.testing.assert_array_equal(.5*(old['raw_query_only']+raw),sf['six_query_equal_mean'])
        assert interval(final,old['extra_real'],test)==run['real_vs_query_query_nmse_interval']
        ids=np.unique([0,len(test['sid'])//2,len(test['sid'])-1]);small={k:v[ids] for k,v in test.items()}
        for model,seed in zip(models,contract['seeds']):np.testing.assert_allclose(predict(model,small,stats,meta['period'],'query_only',device),sf[f'seed_{seed}'][ids],rtol=2e-5,atol=2e-4)
    r=dict(dataset=folder.name,complete=True,query_case_networks=6,query_query_networks=6,same_parameters_and_training_budget=True,policy_and_predictions_rebuilt=True,device=device)
    print('VERIFIED CAPACITY',r,flush=True);return r


def verify(folder,device='cpu',live=False):
    run=read(folder/'run.json');meta=read(folder/'preparation.json');contract=read(folder/'contract.json')
    for file,key in [('predictions.npz','predictions_sha256'),('seed_predictions.npz','seed_predictions_sha256'),('contract.json','contract_sha256')]:
        assert sha256(folder/file)==run[key]
    assert sha256(folder/'validation_predictions.npz')==contract['validation_sha256']
    values={split:trace(folder,split) for split in ['train','validation','test']};stats=read(folder/'stats.json')
    store=Store(meta['store']);pairs=0;matching_checks=0
    try:
        assert store.meta['data_id']==meta['source_data_id']
        for s in store.series:
            sid=s['sid'];raw=store.values(sid);path=folder/'banks'/f'{sid}.npz'
            assert sha256(path)==meta['banks'][str(sid)]
            with np.load(path,allow_pickle=False) as f:bank={k:f[k].copy() for k in f.files}
            np.testing.assert_array_equal(bank['past'],raw[bank['starts'][:,None]+np.arange(96)])
            assert np.all(bank['starts']+192<=int(meta['limits'][str(sid)]))
            bf=state_features(bank['past'],bank['starts'],meta['period'],stats['memory_mean'][sid],stats['memory_std'][sid])
            for split,data in values.items():
                rows=np.flatnonzero(data['sid']==sid);start=data['start'][rows]
                np.testing.assert_array_equal(data['raw_x'][rows],raw[start[:,None]+np.arange(96)])
                np.testing.assert_array_equal(data['raw_y'][rows],raw[start[:,None]+96+np.arange(96)])
                if split=='train':assert np.all(start+192<=s['train_end'])
                else:assert np.all(start>=s['train_end'] if split=='validation' else start>=s['validation_end'])
                for prefix in ['', 'random_']:
                    selected=data[prefix+'selected_starts'][rows]
                    assert np.all(selected+192<=np.minimum(start[:,None],int(meta['limits'][str(sid)])))
                    assert all(len(np.unique(t))==10 for t in selected)
                    assert np.all(np.diff(np.sort(selected[:,5:],axis=1),axis=1)>=192)
                    np.testing.assert_array_equal(data[prefix+'case_past'][rows],raw[selected[:,:,None]+np.arange(96)])
                    np.testing.assert_array_equal(data[prefix+'case_future'][rows],raw[selected[:,:,None]+96+np.arange(96)])
                    ids=np.searchsorted(bank['starts'],selected);np.testing.assert_array_equal(bank['starts'][ids],selected)
                    np.testing.assert_array_equal(data[prefix+'case_baseline'][rows],bank['baseline'][ids])
                    pairs+=len(rows)*10
                assert all(not np.isin(data['random_selected_starts'][r],data['selected_starts'][r]).any() for r in rows)
                for j in [0,5]:np.testing.assert_array_equal(np.sort(data['shuffle_order'][rows,j:j+5],axis=1),np.tile(np.arange(j,j+5),(len(rows),1)))
                for row in np.unique([rows[0],rows[len(rows)//2],rows[-1]]):
                    seed=20261015+sid*1000003+int(data['start'][row])
                    rebuilt=matched_starts(bank['starts'],bank['past'],data['selected_starts'][row],int(data['start'][row]),meta['period'],stats['memory_mean'][sid],stats['memory_std'][sid],seed,bf)
                    np.testing.assert_array_equal(rebuilt,data['random_selected_starts'][row]);matching_checks+=1
            tr=values['train'];va=values['validation'];assert np.max(tr['start'][tr['sid']==sid]+192)<=np.min(va['start'][va['sid']==sid])
    finally:store.close()
    p=UnifiedResidualPredictor(folder,device,None if live else False)
    try:
        selection=p.selection;assert len({r['parameters'] for r in selection['models']})==1
        assert len(selection['models'])==len(MODES)*3
        for r in selection['models']:assert len(r['losses'])==selection['settings']['epochs']==16
        fc=selection['forward_check'];assert fc['last_fitted_end']<=fc['cutoff']
        val=values['validation'];test=values['test'];policies=p.policies
        with np.load(folder/'validation_predictions.npz',allow_pickle=False) as vf,np.load(folder/'seed_predictions.npz',allow_pickle=False) as sf,np.load(folder/'predictions.npz',allow_pickle=False) as saved:
            q,policy=confirmed(val,vf['query_only'],p.stats,val['raw_quantiles'][:,4]);assert policy==policies['query_only']
            np.testing.assert_array_equal(q,vf['query_confirmed'])
            for mode in MODES:
                rebuilt=predict(p.models[mode][0],val,p.stats,p.meta['period'],mode,device)
                # The first model participates in a three-seed average, not alone.
                assert np.isfinite(rebuilt).all()
                raw=np.mean([sf[f'{mode}_{seed}'] for seed in selection['settings']['seeds']],axis=0).astype(np.float32)
                np.testing.assert_array_equal(raw,saved['raw_'+mode])
                for seed in selection['settings']['seeds']:assert metric(sf[f'{mode}_{seed}'],test)==run['individual_seed_metrics'][f'{mode}_{seed}']
                vp=np.mean([predict(m,val,p.stats,p.meta['period'],mode,device) for m in p.models[mode]],0).astype(np.float32)
                np.testing.assert_allclose(vp,vf[mode],rtol=2e-5,atol=2e-4)
                if mode!='query_only':
                    _,own=confirmed(val,vf[mode],p.stats,val['raw_quantiles'][:,4]);assert own==policies['own_'+mode]
                    _,extra=confirmed(val,vf[mode],p.stats,q);assert extra==policies['extra_'+mode]
                    np.testing.assert_array_equal(apply_confirmed(test,raw,saved['moirai_direct'],own),saved['own_'+mode])
                    np.testing.assert_array_equal(apply_confirmed(test,raw,saved['query_confirmed'],extra),saved['extra_'+mode])
            np.testing.assert_array_equal(apply_confirmed(test,saved['raw_query_only'],saved['moirai_direct'],policy),saved['query_confirmed'])
            transfer=fixed_transfer(val,p.stats,p.meta['period']);np.testing.assert_array_equal(transfer,vf['transfer_only'])
            _,tp=confirmed(val,transfer,p.stats,val['raw_quantiles'][:,4]);assert tp==policies['transfer_only']
            np.testing.assert_array_equal(fixed_transfer(test,p.stats,p.meta['period']),saved['fixed_transfer'])
            np.testing.assert_array_equal(apply_confirmed(test,saved['fixed_transfer'],saved['moirai_direct'],tp),saved['confirmed_transfer'])
            for key,metrics in run['metrics'].items():assert metric(saved[key],saved)==metrics
            assert interval(saved['query_confirmed'],saved['extra_real'],saved)==run['additional_rag_nmse_interval']
            active=np.flatnonzero(np.isin(test['sid'],run['active_rag_variables']))
            ids=np.unique([0,int(active[0]) if len(active) else len(test['sid'])//2,len(test['sid'])-1])
            for row in ids:
                sid,start=int(test['sid'][row]),int(test['start'][row])
                output=p.predict(test['raw_x'][row],sid,start) if live else p.predict_cached(test['raw_x'][row],sid,start,test['raw_quantiles'][row:row+1],test['probe_quantiles'][row:row+1])
                for key,forecast in output['forecasts'].items():np.testing.assert_allclose(forecast,saved[key][row],rtol=3e-5,atol=3e-4)
                np.testing.assert_array_equal(output['selected_starts'],test['selected_starts'][row]);np.testing.assert_array_equal(output['random_starts'],test['random_selected_starts'][row])
            if live:
                adapter=p.previous.previous.original.adapter;sid=int(test['sid'][0]);bank=p.bank(sid);ix=np.unique([0,len(bank['starts'])-1])
                old=adapter.forecast(bank['past'][ix],96)[:,4].cpu().numpy();np.testing.assert_allclose(old,bank['baseline'][ix],rtol=3e-5,atol=3e-4)
            row=int(ids[0]);sid,start=int(test['sid'][row]),int(test['start'][row]);q0=test['raw_quantiles'][row:row+1];probe=test['probe_quantiles'][row:row+1]
            before=p.predict_cached(test['raw_x'][row],sid,start,q0,probe)
            for branch in p.previous.previous.branches.values():
                for raw in branch.synchronous_values:raw[start+96:start+192]=1e15
            after=p.predict_cached(test['raw_x'][row],sid,start,q0,probe);assert before==after
    finally:p.close()
    record=dict(dataset=folder.name,queries=len(values['test']['sid']),case_pairs=pairs,matching_rule_replays=matching_checks,
        physical_cases_targets_baselines=True,full_policy_metric_reconstruction=True,actual_deployment_queries=len(ids),
        query_and_sibling_future_poison_invariant=True,actual_gpu=live,reference='Identical pinned frozen Moirai; no dataset fit',
        upstream_training_retrieval_is_forward_oof=False)
    print('VERIFIED UNIFIED',record,flush=True);return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--live',action='store_true');g=p.add_mutually_exclusive_group();g.add_argument('--capacity-only',action='store_true');g.add_argument('--derangement-only',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    records=[verify_derangement(Path(a.root)/name,'cuda' if a.live else 'cpu') if a.derangement_only else (verify_capacity(Path(a.root)/name,'cuda' if a.live else 'cpu') if a.capacity_only else verify(Path(a.root)/name,'cuda' if a.live else 'cpu',a.live)) for name in a.datasets]
    filename=('gpu_' if a.live else '')+('derangement_verification.json' if a.derangement_only else ('capacity_verification.json' if a.capacity_only else 'verification.json'));out=Path(a.root)/filename
    previous=read(out)['records'] if out.exists() else [];merged={r['dataset']:r for r in previous};merged.update({r['dataset']:r for r in records})
    write_json(out,dict(complete=True,records=list(merged.values())))
