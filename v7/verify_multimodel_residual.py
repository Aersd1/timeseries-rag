"""Physical boundaries, primary losses, exact-df probes and actual GPU spot checks."""
import argparse,gc
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from .common import sha256,write_json
from .benchmark_memory_access import read
from .benchmark_multimodel_residual import arrays,model_path,restore,losses
from .multimodel_residual import (L,H,PHASES,ARMS,materialize,feature_blocks,apply_probe,predict,norm,fit_probe,query_metadata)
from .moirai import MoiraiAdapter,MODEL_ID,MODEL_REVISION


def verify(root,device='cuda'):
    s=read(root/'summary.json');p=s['protocol'];frozen=read(root/'frozen.json')
    if not s['complete']:raise ValueError('Incomplete matrix')
    for file,h in p['code_sha256'].items():
        if sha256(file)!=h:raise AssertionError('Changed experimental source')
    for file,h in frozen.items():
        if sha256(file)!=h:raise AssertionError('Changed frozen development artifact')
    plan=read(root/'analysis_plan.json')
    if not plan['written_before_F_scoring'] or sha256('v7/analyze_multimodel_residual.py')!=plan['analysis_sha256']:raise AssertionError('Changed pre-F analysis')
    torch.set_num_threads(4)
    foundation=MoiraiAdapter.pretrained(dict(model_id=MODEL_ID,revision=MODEL_REVISION,
        cache_dir='.moirai-cache',local_files_only=True),device)
    physical_windows=0;probe_predictions=0;base_predictions=0;reconstructed=0;refits=0;naive_quality={}
    for name in p['datasets']:
        folder=root/name;packet=read(folder/'packet.json');store=Store(packet['store'])
        try:
            data={phase:arrays(folder/f'data_{phase}.npz') for phase in ('A','B','C','D','E')}
            for sid in packet['sids']:
                series=store.series[sid];raw=store.values(sid)
                if sha256(store.path/series['file'])!=packet['values_sha256'][str(sid)]:raise AssertionError('Changed physical series')
                train=np.asarray(raw[:int(.32*series['n'])]);finite=train[np.isfinite(train)]
                np.testing.assert_allclose(packet['stats'][str(sid)]['std'],max(float(finite.std()),1e-6),rtol=1e-12)
                for phase,d in data.items():
                    rows=np.flatnonzero(d['sid']==sid);starts=d['start'][rows];lo,hi,_,_=PHASES[phase]
                    if np.any(starts<int(lo*series['n'])) or np.any(starts+L+H>int(hi*series['n'])):raise AssertionError('Crossed temporal role')
                    if np.any(starts%PHASES[phase][2]):raise AssertionError('Phase not on global time grid')
                    np.testing.assert_array_equal(d['x'][rows],raw[starts[:,None]+np.arange(L)])
                    np.testing.assert_array_equal(d['y'][rows],raw[starts[:,None]+L+np.arange(H)])
                    physical_windows+=len(rows)
            data['D']=query_metadata(data['D'],packet)
            bank=data['C'];test_refs=arrays(folder/'test_refs.npz')['refs'];test=query_metadata(materialize(store,test_refs,packet['stats']),packet)
            for sid in packet['sids']:
                rows=test['sid']==sid;np.testing.assert_array_equal(test['start'][rows],np.unique(test['start']))
                if np.any(test['start'][rows]<int(.8*store.series[sid]['n'])):raise AssertionError('TEST before80%')
            physical_windows+=len(test_refs)
            period=int(round(packet['daily_period']))
            naive=np.tile(test['x'][:,-period:],(1,int(np.ceil(H/period))))[:,:H]
            naive_quality[name]=dict(zip(('mse','mae','train_nmse'),losses(naive,test['y'],test['std']).mean(0).tolist()))
            positions=np.unique(np.rint(np.linspace(0,len(test_refs)-1,9)).astype(int))
            sample={k:v[positions] for k,v in test.items()};seen=set()
            for kind in p['models']:
                for seed in p['seeds']:
                    path=model_path(folder,kind,seed);record=read(path/f'test_{seed}.json');rowfile=path/f'test_rows_{seed}.npz'
                    if sha256(rowfile)!=record['rows_sha256']:raise AssertionError('Changed TEST rows')
                    saved=arrays(rowfile)
                    for key in ('sid','start'):np.testing.assert_array_equal(saved[key],test[key])
                    np.testing.assert_array_equal(saved['raw_y'],test['y'])
                    for arm,key in [('base','base'),('query_only','query_only'),('real','real')]:
                        v=losses(saved[key],test['y'],test['std']);np.testing.assert_allclose(saved['loss_'+arm],v,rtol=1e-12,atol=1e-10);reconstructed+=len(v)
                    for arm in ('base',)+ARMS:
                        value=saved['loss_'+arm].mean(0)
                        np.testing.assert_allclose(value,[record['metrics'][arm][k] for k in ('mse','mae','train_nmse')],rtol=1e-12,atol=1e-10)
                    cases=arrays(folder/f'cases_F_{seed}.npz')
                    for prefix in ('real','random'):
                        ids=cases[prefix]
                        if np.any(bank['start'][ids]+L+H>test['start'][:,None]):raise AssertionError('Self/overlapping retrieval')
                        if any(len(np.unique(row))!=10 for row in ids):raise AssertionError('Duplicated5+5')
                    if any(np.intersect1d(a,b).size for a,b in zip(cases['real'],cases['random'])):raise AssertionError('Random arm reuses real cases')
                    starts=bank['start'][cases['real'][:,5:]]
                    difference=abs(starts[:,:,None]-starts[:,None,:])+np.eye(5,dtype=np.int64)[None]*10**12
                    if np.any(difference<192):raise AssertionError('Raw-history5 not diverse')
                    bases=arrays(path/'base_predictions.npz')
                    if str(path) not in seen:
                        model=restore(path,kind,device,foundation)
                        np.testing.assert_allclose(predict(model,kind,sample,device),saved['base'][positions],rtol=2e-5,atol=3e-4)
                        ci=np.unique(np.rint(np.linspace(0,len(bank['x'])-1,3)).astype(int));bc={k:v[ci] for k,v in bank.items()}
                        np.testing.assert_allclose(predict(model,kind,bc,device),bases['C'][ci],rtol=2e-5,atol=3e-4)
                        base_predictions+=len(positions)+len(ci);seen.add(str(path))
                        if kind not in ('ridge','moirai2'):del model;torch.cuda.empty_cache()
                    for arm in ARMS:
                        probe=arrays(path/f'probe_{seed}_{arm}.npz')
                        if abs(float(probe['effective_df'])-min(64.,len(data['D']['x'])/2))>.05:raise AssertionError('Mismatched effective complexity')
                        common,extra,scale=feature_blocks(sample,saved['base'][positions],bank,bases['C'],cases['real'][positions],cases['random'][positions],arm,seed)
                        result=apply_probe(probe,common,extra,saved['base'][positions],scale)
                        expected=losses(result,sample['y'],sample['std'])
                        np.testing.assert_allclose(saved['loss_'+arm][positions],expected,rtol=1e-8,atol=1e-7);probe_predictions+=len(positions)
                        # Refit the two decisive predictor arms on ALL D rows.
                        if arm in ('query_only','real'):
                            di=arrays(folder/f'cases_D_{seed}.npz');d=data['D']
                            co,ex,sc=feature_blocks(d,bases['D'],bank,bases['C'],di['real'],di['random'],arm,seed)
                            target=(d['y']-bases['D'])/sc[:,None];rebuilt=fit_probe(co,ex,target,seed)
                            for key in probe:np.testing.assert_allclose(probe[key],rebuilt[key],rtol=1e-7,atol=1e-8)
                            refits+=1
                    if (root/'frozen.json').stat().st_mtime_ns>rowfile.stat().st_mtime_ns:raise AssertionError('TEST before freeze')
            print('MULTI VERIFIED',name,flush=True)
        finally:store.close()
        gc.collect()
    report=dict(passed=True,physical_windows=physical_windows,primary_loss_rows_reconstructed=reconstructed,
        actual_backbone_forecasts_spot_checked=base_predictions,all_arm_probe_forecasts_spot_checked=probe_predictions,
        full_D_query_and_real_probe_refits=refits,
        seasonal_naive_quality=naive_quality,
        scope='All physical development/TEST windows and temporal roles, source/freeze hashes, full primary loss reconstruction, all-arm loss reductions, checkpoint and probe spot inference; full refits of Q/R, not every negative-control fit')
    write_json(root/'verification.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    print(verify(Path(a.root),a.device))
