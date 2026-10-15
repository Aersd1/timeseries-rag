"""Audit physical time boundaries, prefix states, held predictions and routing."""
import argparse,gc
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from .common import sha256,write_json
from .dense_forward import read,materialize,prefix_state,PrefixRetriever
from .forward_experts import ARMS,model_from
from .benchmark_unified_residual import predict as expert_predict
from .benchmark_forward_selectors import development,history_data,load_models,load_npz
from .forward_selectors import FAMILIES,SETTINGS,inputs,labels,prediction,choose_blocks,calibration,apply_policy,portfolio
from .selective_rag import error_metrics


def verify_utilities(actual,expected,family,device):
    difference=np.abs(np.asarray(actual,dtype=float)-expected)
    relative=difference/(1+np.abs(expected))
    summary=dict(max_absolute=float(difference.max()),rms_scaled=float(np.sqrt(np.mean(relative**2))),
        q999_scaled=float(np.quantile(relative,.999)),max_scaled=float(relative.max()))
    if family=='raw_sequence' and device=='cpu':
        # Different CPU/CUDA attention kernels can amplify roundoff in a few
        # outlier scores. Bound the full error distribution and expose route
        # changes below; do not claim bit-identical native device decisions.
        if not (summary['rms_scaled']<1e-4 and summary['q999_scaled']<1e-3 and summary['max_scaled']<5e-3):
            raise ValueError('CPU sequence utility drift exceeds portability bounds: '+str(summary))
    else:np.testing.assert_allclose(actual,expected,rtol=5e-5,atol=5e-5)
    return summary


def verify_provenance(out):
    meta=read(out/'preparation.json');contract=read(out/'selector_contract.json');source=Path(meta['source'])
    if sha256(source/'preparation.json')!=meta['source_preparation_sha256']:raise ValueError('Changed source preparation')
    if sha256(Path(meta['store'])/'catalog.json')!=meta['catalog_sha256']:raise ValueError('Changed data catalog')
    if sha256(out/f'prefix_{meta["layout"]["cuts"][0]}.json')!=contract['provenance']['anchor_sha256']:raise ValueError('Changed anchor statistics')
    checked=0
    for phase,expected in contract['provenance']['phases'].items():
        if sha256(out/(phase+'_experts.json'))!=expected:raise ValueError('Changed frozen upstream phase')
        r=read(out/(phase+'_experts.json'))
        for name,key in [(phase+'_predictions.npz','predictions_sha256'),(phase+'_train_index.npz','train_index_sha256'),
            (phase+'_held_index.npz','held_index_sha256'),(f'prefix_{r["cutoff"]}.json','stats_sha256')]:
            if sha256(out/name)!=r[key]:raise ValueError('Changed upstream '+name)
            checked+=1
    for namespace,manifest in [('foundation','foundation_manifest.json'),('foundation_test','test_foundation_manifest.json')]:
        for sid,expected in read(out/manifest)['files'].items():
            if sha256(out/namespace/(sid+'.npz'))!=expected:raise ValueError('Changed foundation file '+namespace+'/'+sid)
            checked+=1
    for name,expected in contract['models'].items():
        path=out/'selectors'/(name+'.json');entry=read(path)
        if sha256(path)!=expected or sha256(path.parent/entry['file'])!=entry['sha256']:raise ValueError('Changed selector '+name)
        if entry['provenance']!=contract['provenance']:raise ValueError('Mismatched selector training provenance '+name)
        if [r['seed'] for r in entry['records']]!=SETTINGS['seeds']:raise ValueError('Mismatched selector seeds')
        for r in entry['records']:
            if entry['family']=='hist_gbdt':assert r['iterations']==SETTINGS['tree_iterations']
            else:assert len(r['losses'])==SETTINGS['epochs']
        checked+=2
    return dict(dataset=out.name,complete=True,hashed_files=checked)


def verify_prefixes(source,out):
    meta=read(out/'preparation.json');store=Store(meta['store']);count=0;replayed=0
    try:
        for s in store.series:
            if sha256(Path(meta['store'])/s['file'])!=s['values_sha256']:raise ValueError('Changed physical dataset')
        selected_sids=sorted({0,len(store.series)//2,len(store.series)-1}|({313} if out.name=='electricity' else set())|({8} if out.name=='weather' else set()))
        for phase in ['fold0','fold1','fold2','final']:
            r=read(out/(phase+'_experts.json'));cut=r['cutoff'];stats=read(out/f'prefix_{cut}.json')
            if prefix_state(store,cut)!=stats:raise ValueError('Prefix fit statistics/groups differ')
            index={}
            for role in ['train','held']:
                name=phase+'_'+role;info=read(out/(name+'_index.json'));path=out/(name+'_index.npz')
                if sha256(path)!=info['sha256']:raise ValueError('Changed index')
                d=load_npz(path);index[role]=d
                if role=='train':assert np.all(d['start']+192<=cut)
                else:assert np.all(d['start']>=cut)
                for key in ['selected_starts','random_selected_starts']:
                    limit=np.array([min(store.series[int(s)]['memory_end'],cut) for s in d['sid']])
                    assert np.all(d[key]+192<=np.minimum(d['start'],limit)[:,None]);count+=d[key].size
            for sid in selected_sids:
                with np.load(source/'banks'/f'{sid}.npz',allow_pickle=False) as bank:stride=int(np.diff(bank['starts']).min())
                retriever=PrefixRetriever(store,stats,sid,meta['period'],stride)
                with np.load(out/'foundation'/f'{sid}.npz',allow_pickle=False) as f:cached={int(t):q for t,q in zip(f['starts'],f['quantiles'])}
                for role,d in index.items():
                    rows=np.flatnonzero(d['sid']==sid)
                    for row in rows[np.unique(np.linspace(0,len(rows)-1,min(2,len(rows)),dtype=int))]:
                        start=int(d['start'][row]);a,b=retriever.select(store.window(sid,start,96),start,cached[start])
                        np.testing.assert_array_equal(a,d['selected_starts'][row]);np.testing.assert_array_equal(b,d['random_selected_starts'][row]);replayed+=1
            for record in r['models']:
                if sha256(out/record['file'])!=record['sha256']:raise ValueError('Changed prefix expert')
                assert len(record['losses'])==16
    finally:store.close()
    return dict(case_pairs=count,retrieval_replays=replayed)


def verify(out,device='cpu',prefixes=True):
    meta=read(out/'preparation.json');source=Path(meta['source']);contract=read(out/'selector_contract.json');run=read(out/'run.json');policies=read(out/'selector_policy.json')
    provenance=verify_provenance(out)
    for file,expected in [('selector_contract.json',run['contract_sha256']),('selector_policy.json',contract['policy_sha256']),
        ('selector_validation.npz',contract['validation_sha256']),('predictions.npz',run['predictions_sha256']),
        ('routing.npz',run['routing_sha256']),('utility_predictions.npz',run['utility_sha256'])]:
        if sha256(out/file)!=expected:raise ValueError('Changed '+file)
    for file,expected in contract['code_sha256'].items():
        if sha256('v7/'+file)!=expected:raise ValueError('Changed frozen implementation '+file)
    if sha256(out/'foundation_manifest.json')!=contract['provenance']['foundation_manifest_sha256']:raise ValueError('Changed developmental forecasts')
    audit=verify_prefixes(source,out) if prefixes else {}
    train,tp,val,vp=development(source,out);d=materialize(source,out,'test');anchor=read(out/f'prefix_{meta["layout"]["cuts"][0]}.json');final=read(out/f'prefix_{meta["layout"]["cuts"][-1]}.json')
    saved=load_npz(out/'predictions.npz');utility=load_npz(out/'utility_predictions.npz');routing=load_npz(out/'routing.npz');valutility=load_npz(out/'selector_validation.npz')
    for k in ['sid','start','raw_y','std']:np.testing.assert_array_equal(saved[k],d[k])
    assert np.all(d['selected_starts']+192<=d['start'][:,None]);assert np.all(d['random_selected_starts']+192<=d['start'][:,None])
    hist=history_data(train);vhist=history_data(train,val);testhist=history_data(train,val)
    blocks=choose_blocks(val,meta['layout']['validation_end']);assert contract['calibration_origins']==[np.unique(val['start'][b]).tolist() for b in blocks]
    assert max(val['start'])+192<=min(d['start'])
    cohorts=dict(forward_full=np.arange(len(train['sid'])),sparse_last4=np.flatnonzero(np.isin(train['start'],np.unique(train['start'])[-4:])))
    records=[]
    for arm in ARMS:
        model=model_from(torch.load(out/f'final_{arm}.pt',map_location='cpu',weights_only=True),device)
        native=expert_predict(model,d,final,meta['period'],arm,device);np.testing.assert_allclose(native,saved[arm+'__raw'],rtol=4e-5,atol=4e-5)
        candidate=saved[arm+'__raw'];del model
        hp=np.concatenate([tp[arm],vp[arm]]);gain,harm=labels(train,tp[arm],anchor)
        xp=inputs(d,candidate,anchor,meta['period'],arm,testhist,hp,raw=True)
        poisoned=dict(d,raw_y=np.full_like(d['raw_y'],np.nan));xa=inputs(poisoned,candidate,anchor,meta['period'],arm,testhist,hp,raw=True)
        for k in xp:np.testing.assert_array_equal(xp[k],xa[k])
        del xa
        xt=inputs(train,tp[arm],anchor,meta['period'],arm,hist,tp[arm],raw=False)
        xvraw=inputs(val,vp[arm],anchor,meta['period'],arm,vhist,hp,raw=True)
        for family in FAMILIES:
            xv=xvraw if family=='raw_sequence' else {'summary':xvraw['summary']}
            xx=xp if family=='raw_sequence' else {'summary':xp['summary']}
            for cohort,ids in cohorts.items():
                key=cohort+'__'+arm+'__'+family;path=out/'selectors'/key;entry=read(path.with_suffix('.json'))
                if sha256(path.with_suffix('.json'))!=contract['models'][key]:raise ValueError('Changed selector manifest')
                models,normal=load_models(path,device);flat=xt['summary'][ids].reshape(-1,52).astype(float)
                np.testing.assert_allclose(normal['mean'],flat.mean(0),rtol=0,atol=1e-10)
                np.testing.assert_allclose(normal['scale'],np.maximum(flat.std(0),.05),rtol=0,atol=1e-10)
                np.testing.assert_allclose(normal['target_scale'],max(float(np.sqrt(np.mean(gain[ids]**2))),.01),rtol=0,atol=1e-10)
                val_pred=prediction(family,models,normal,xv,device);val_drift=verify_utilities(val_pred,valutility[key],family,device)
                assert calibration(val,vp[arm],valutility[key],family,anchor,blocks)==policies[key]
                pred=prediction(family,models,normal,xx,device);test_drift=verify_utilities(pred,utility[key],family,device)
                # Replay exact routing from frozen utility predictions; native scores
                # have independently been checked above on the requested device.
                forecast,mask=apply_policy(d,candidate,utility[key],policies[key]);np.testing.assert_array_equal(forecast,saved[key]);np.testing.assert_array_equal(mask,routing[key])
                np.testing.assert_array_equal(forecast.reshape(-1,12,8)[~mask],d['raw_quantiles'][:,4].reshape(-1,12,8)[~mask])
                for scope in ['dense','legacy']:
                    keep=saved[scope+'_mask'];truth={k:d[k][keep] for k in ['raw_y','sid','start','std']};m=error_metrics(truth,forecast[keep],anchor)
                    for k,v in m.items():np.testing.assert_allclose(v,run['metrics'][scope][key][k],rtol=1e-12,atol=1e-12)
                native_forecast,native_mask=apply_policy(d,candidate,pred,policies[key])
                records.append(dict(key=key,utility_max_difference=float(np.max(abs(pred-utility[key]))),
                    validation_utility_max_difference=float(np.max(abs(val_pred-valutility[key]))),
                    native_route_differences=int(np.sum(native_mask!=mask)),
                    native_forecast_max_difference=float(np.max(abs(native_forecast-forecast))),
                    validation_numeric_drift=val_drift,test_numeric_drift=test_drift,
                    future_invariant=True,exact_fallback=True))
                del models,pred,val_pred;gc.collect()
                print('FORWARD VERIFIED',out.name,key,device,flush=True)
            del xv
        static=calibration(val,vp[arm],None,'static',anchor,blocks);assert static==policies[arm+'__static']
        for cohort in cohorts:
            key=cohort+'__'+arm;pool={f:policies[key+'__'+f] for f in FAMILIES};pool['static']=static
            chosen=portfolio(pool);assert chosen==policies[key+'__portfolio']
            pp=None if chosen['selected_family']=='static' else utility[key+'__'+chosen['selected_family']]
            forecast,mask=apply_policy(d,candidate,pp,chosen);np.testing.assert_array_equal(forecast,saved[key+'__portfolio']);np.testing.assert_array_equal(mask,routing[key+'__portfolio'])
        del xp,xt,xvraw;gc.collect()
    # Legacy Moirai and targets must match the previous benchmark exactly.
    old=load_npz(source/'test.npz')
    for sid in np.unique(d['sid']):
        oldrows=np.flatnonzero(old['sid']==sid);newrows=np.flatnonzero((d['sid']==sid)&saved['legacy_mask'])
        np.testing.assert_array_equal(d['start'][newrows],old['start'][oldrows]);np.testing.assert_array_equal(saved['moirai_direct'][newrows],old['raw_quantiles'][oldrows,4]);np.testing.assert_array_equal(d['raw_y'][newrows],old['raw_y'][oldrows])
    return dict(dataset=out.name,device=device,torch_version=torch.__version__,complete=True,records=records,provenance=provenance,prefix_audit=audit,queries=len(d['sid']),dense_queries=run['dense_queries'],legacy_queries=run['legacy_queries'])


def verify_deployment(out,device='cuda'):
    from .query_forward_selectors import ForwardSelectorPredictor,ObservedStore
    saved=load_npz(out/'predictions.npz');index=load_npz(out/'test_index.npz')
    predictor=ForwardSelectorPredictor(out,device,adapter=False);key=read(out/'run.json')['primary'];records=[]
    sids=sorted({0,int(saved['sid'].max())//2,int(saved['sid'].max())}|({313} if out.name=='electricity' else set())|({8} if out.name=='weather' else set()))
    try:
        for sid in sids:
            ids=np.flatnonzero(saved['sid']==sid)
            with np.load(out/'foundation_test'/f'{sid}.npz',allow_pickle=False) as f:cache={int(t):(q,p) for t,q,p in zip(f['starts'],f['quantiles'],f['probe'])}
            for row in ids[np.unique([0,len(ids)-1])]:
                start=int(saved['start'][row]);x=predictor.store.window(sid,start,96);q,probe=cache[start]
                result=predictor.predict_cached(x,sid,start,q[None],probe[None])
                np.testing.assert_array_equal(result['selected_starts'],index['selected_starts'][row])
                np.testing.assert_allclose(result['candidate_forecast'],saved['real__raw'][row],rtol=5e-5,atol=5e-5)
                np.testing.assert_allclose(result['forecast'],saved[key][row],rtol=5e-5,atol=5e-5)
                original=predictor.store;predictor.store=ObservedStore(original,start+96)
                try:limited=predictor.predict_cached(x,sid,start,q[None],probe[None])
                finally:predictor.store=original
                np.testing.assert_array_equal(result['forecast'],limited['forecast'])
                np.testing.assert_array_equal(result['selected_starts'],limited['selected_starts'])
                records.append(dict(sid=sid,start=start,selected_cases_exact=True,physically_future_limited=True,
                    forecast_max_difference=float(np.max(abs(np.asarray(result['forecast'])-saved[key][row])))))
                print('FORWARD DEPLOYMENT',out.name,sid,start,flush=True)
    finally:predictor.close()
    return dict(dataset=out.name,complete=True,device=device,records=records)


def intervene(out,device='cuda'):
    """Diagnostic ONLY: freeze real weights, policy strength and adoption mask.

    No new TEST-selected policy is constructed. Removing values can shift the
    input distribution; this complements, rather than replaces, retrained arms.
    """
    from .forward_diagnostics import compare
    result_file=out/'reference_interventions.json';pred_file=out/'reference_interventions.npz'
    implementation={name:sha256('v7/'+name) for name in ['verify_forward_selectors.py','forward_diagnostics.py']}
    if result_file.exists():
        result=read(result_file)
        if result['implementation']!=implementation or sha256(pred_file)!=result['predictions_sha256']:raise ValueError('Changed intervention replay')
        return result
    meta=read(out/'preparation.json');run=read(out/'run.json');key=run['primary'];policies=read(out/'selector_policy.json');policy=policies[key]
    if sha256(out/'selector_contract.json')!=run['contract_sha256'] or sha256(out/'routing.npz')!=run['routing_sha256']:raise ValueError('Changed frozen routing contract')
    contract=read(out/'selector_contract.json')
    if sha256(out/'selector_policy.json')!=contract['policy_sha256'] or sha256(out/'predictions.npz')!=run['predictions_sha256']:raise ValueError('Changed frozen policy/predictions')
    source=Path(meta['source']);d=materialize(source,out,'test');anchor=read(out/f'prefix_{meta["layout"]["cuts"][0]}.json');stats=read(out/f'prefix_{meta["layout"]["cuts"][-1]}.json')
    entry=next(r for r in read(out/'final_experts.json')['models'] if r['mode']=='real');checkpoint=out/entry['file']
    if sha256(checkpoint)!=entry['sha256']:raise ValueError('Changed real expert')
    model=model_from(torch.load(checkpoint,map_location='cpu',weights_only=True),device)
    with np.load(out/'predictions.npz',allow_pickle=False) as f:
        original_raw=f['real__raw'];original=f[key];keep=f['dense_mask'];base=f['moirai_direct']
        for k in ['sid','start','raw_y']:np.testing.assert_array_equal(f[k],d[k])
    with np.load(out/'routing.npz',allow_pickle=False) as f:mask=f[key]
    np.testing.assert_array_equal(base,d['raw_quantiles'][:,4]);truth={k:d[k][keep] for k in ['raw_y','sid','start','std']}
    forecasts={};records={}
    for name in ['without_values','deranged_values','matched_random']:
        modified=dict(d,case_future=d['case_baseline']) if name=='without_values' else d
        mode={'without_values':'real','deranged_values':'shuffled','matched_random':'matched_random'}[name]
        raw=expert_predict(model,modified,stats,meta['period'],mode,device)
        selected=base.copy();view=selected.reshape(-1,12,8);proposal=(base+policy['strength']*(raw-base)).reshape(-1,12,8);view[mask]=proposal[mask]
        np.testing.assert_array_equal(view[~mask],base.reshape(-1,12,8)[~mask])
        forecasts[name+'__raw']=raw;forecasts[name+'__fixed_route']=selected
        records[name]=dict(raw_metrics=error_metrics(truth,raw[keep],anchor),fixed_route_metrics=error_metrics(truth,selected[keep],anchor),
            original_raw_relative_to_intervention=compare(raw[keep],original_raw[keep],d['raw_y'][keep],d['start'][keep],meta['period']),
            original_selected_relative_to_intervention=compare(selected[keep],original[keep],d['raw_y'][keep],d['start'][keep],meta['period']),
            raw_max_change=float(np.max(abs(raw-original_raw))),selected_max_change=float(np.max(abs(selected-original))))
        print('FROZEN REFERENCE INTERVENTION',out.name,name,flush=True)
    np.savez_compressed(pred_file,**forecasts,sid=d['sid'],start=d['start'],dense_mask=keep)
    result=dict(dataset=out.name,complete=True,device=device,implementation=implementation,adopted_fraction=float(mask[keep].mean()),
        primary=key,records=records,model_sha256=entry['sha256'],contract_sha256=run['contract_sha256'],routing_sha256=run['routing_sha256'],predictions_sha256=sha256(pred_file),
        limitation='Post-hoc fixed-weight/fixed-route diagnostic; null-value distribution shift and original-mask conditioning preclude a complete causal claim. No parameter tuning or primary-output replacement.')
    write_json(result_file,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--device',default='cpu');p.add_argument('--skip-prefixes',action='store_true');group=p.add_mutually_exclusive_group();group.add_argument('--deploy-only',action='store_true');group.add_argument('--intervene-only',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    root=Path(a.root);file=root/('intervention_verification.json' if a.intervene_only else 'deployment_verification.json' if a.deploy_only else 'gpu_verification.json' if a.device.startswith('cuda') else 'verification.json');records={x['dataset']:x for x in read(file)['records']} if file.exists() else {}
    for name in a.datasets:
        records[name]=intervene(root/name,a.device) if a.intervene_only else verify_deployment(root/name,a.device) if a.deploy_only else verify(root/name,a.device,not a.skip_prefixes)
        write_json(file,dict(complete=True,records=list(records.values())))
