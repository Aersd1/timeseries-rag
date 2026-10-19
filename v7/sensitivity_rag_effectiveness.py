"""Explicitly POST-HOC controls; do not relabel them as preregistered evidence."""
import argparse
from pathlib import Path
import numpy as np
from .common import write_json
from .benchmark_memory_access import read
from .benchmark_rag_effectiveness import load,interval_losses
from .history_length import metrics
from .rag_effectiveness import FEATURES,ALPHA,transfer


def demean(values,sid,start):
    z=np.asarray(values,dtype=float).copy()
    if z.ndim==1:z=z[:,None]
    # Alternating projections support gapped panels as well as balanced panels.
    for _ in range(20):
        for group in (sid,start):
            _,inverse=np.unique(group,return_inverse=True);count=np.bincount(inverse)
            for column in range(z.shape[1]):
                mean=np.bincount(inverse,weights=z[:,column])/count
                z[:,column]-=mean[inverse]
    return z


def remove_shared_direction(correction,base_shift):
    r=np.asarray(correction,dtype=float);s=np.asarray(base_shift,dtype=float)
    dot=(r*s).sum(1);energy=(s*s).sum(1)
    coefficient=np.divide(np.maximum(dot,0),energy,out=np.zeros(len(r)),where=energy>1e-20)
    projected=r-coefficient[:,None]*s
    norm=(r*r).sum(1);new_norm=(projected*projected).sum(1)
    shrink=np.sqrt(np.divide(new_norm,norm,out=np.zeros(len(r)),where=norm>1e-20))
    return projected,r*shrink[:,None]


def run(root):
    summary=read(root/'summary.json');records=[]
    for record in summary['records']:
        name=record['dataset'];d,evidence=load(name,'test')
        with np.load(root/name/'test_rows.npz',allow_pickle=False) as f:
            fvalues=f['features'].copy();b=f['base_long'].copy();short=f['base_short'].copy();real=f['real_long'].copy()
            loss={k[5:]:f[k].copy() for k in f.files if k.startswith('loss_')}
        with np.load(root/name/'validation_features.npz',allow_pickle=False) as f:
            vfeatures=f['features'].copy();vsid=f['sid'].copy()
        y=d['raw_y'].astype(float);scale=np.maximum(d['raw_x'].std(1),.1*d['train_std'])
        references={'history_mean':np.repeat(d['raw_x'].mean(1)[:,None],96,axis=1),
            'last16_level':np.repeat(d['raw_x'][:,-16:].mean(1)[:,None],96,axis=1),
            'repeat96':d['raw_x'],'short_forecast':short}
        controls={key:b+ALPHA*np.clip(value-b,-2*scale[:,None],2*scale[:,None]) for key,value in references.items()}
        orthogonal,matched=remove_shared_direction(real-b,b-short)
        controls.update(remove_positive_shared_direction=b+orthogonal,norm_matched_shrinkage=b+matched)
        base_mse=((b-y)**2).mean(1);scores={}
        for key,pred in controls.items():
            error=((pred-y)**2).mean(1)
            scores[key]=dict(metrics=metrics(y,pred,d['train_std']),gain_vs_long=interval_losses(base_mse,error,d['start'],evidence['period']))
        projection=((controls['remove_positive_shared_direction']-y)**2).mean(1)
        shrink=((controls['norm_matched_shrinkage']-y)**2).mean(1)
        projection_vs_shrink=interval_losses(shrink,projection,d['start'],evidence['period'])
        np.testing.assert_allclose((orthogonal**2).sum(1),(matched**2).sum(1),rtol=1e-8,atol=1e-7)
        # Apply the IDENTICAL projection to random and each wrong-pair control.
        projected_controls={}
        for arm in ('matched_random','deranged_1','deranged_2','deranged_3','deranged_4'):
            prefix='random_' if arm=='matched_random' else '';parts=[]
            for begin in range(0,len(b),256):
                rows=slice(begin,begin+256)
                p=transfer(d['raw_x'][rows],d['raw_quantiles'][rows],d['probe_quantiles'][rows],
                    d['sid'][rows],d['start'][rows],d[prefix+'case_past'][rows],d[prefix+'case_future'][rows],
                    d[prefix+'case_baseline'][rows],d[prefix+'selected_starts'][rows],d['train_std'][rows],
                    b[rows],evidence['length'],arm=arm,with_features=False)['residual']
                orthogonal_control,_=remove_shared_direction(p-b[rows],b[rows]-short[rows])
                parts.append(((b[rows]+orthogonal_control-y[rows])**2).mean(1))
            projected_controls[arm]=np.concatenate(parts)
        projected_real_vs_random=interval_losses(projected_controls['matched_random'],projection,d['start'],evidence['period'])
        projected_real_vs_deranged=interval_losses(np.mean([projected_controls[f'deranged_{j}'] for j in (1,2,3,4)],0),projection,d['start'],evidence['period'])
        original_b=loss['short_no_rag'][:,2];original_r=loss['original5plus5'][:,2]
        bounded_original=(original_b-original_r)/np.maximum(original_b+original_r,1e-20)
        current_b=loss['long_no_rag'][:,2];current_r=loss['long_residual_real'][:,2]
        bounded_current=(current_b-current_r)/np.maximum(current_b+current_r,1e-20)
        z=demean(np.c_[fvalues,bounded_original,bounded_current],d['sid'],d['start'])
        associations={}
        for j,key in enumerate(FEATURES):
            values={}
            for label,column in [('original_pipeline',-2),('fixed_long_residual',-1)]:
                denom=np.linalg.norm(z[:,j])*np.linalg.norm(z[:,column])
                values[label]=float(z[:,j]@z[:,column]/denom) if denom>1e-12 else None
            associations[key]=values
        # Per-variable VAL thresholds. Only compare columns represented in BOTH
        # TEST tails, with at least2 observations in each; never select on gains.
        within={}
        for j,key in enumerate(FEATURES):
            paired=[]
            for sid in np.unique(d['sid']):
                vi=vsid==sid;ti=d['sid']==sid
                if vi.sum()<4:continue
                lo,hi=np.quantile(vfeatures[vi,j],[.25,.75])
                low=ti&(fvalues[:,j]<=lo);high=ti&(fvalues[:,j]>=hi)
                if low.sum()<2 or high.sum()<2 or hi-lo<1e-10:continue
                pair={}
                for label,ba,ra in [('original',original_b,original_r),('fixed_long',current_b,current_r)]:
                    pair[label]=[float(100*(1-ra[mask].sum()/max(ba[mask].sum(),1e-20))) for mask in (low,high)]
                paired.append(pair)
            within[key]=dict(paired_variables=len(paired),
                mean_low_high_original=np.mean([p['original'] for p in paired],0).tolist() if paired else None,
                mean_low_high_fixed_long=np.mean([p['fixed_long'] for p in paired],0).tolist() if paired else None)
        records.append(dict(dataset=name,query_only_and_direction_controls=scores,
            direction_vs_equal_norm_shrinkage=projection_vs_shrink,
            projected_real_vs_random=projected_real_vs_random,projected_real_vs_deranged=projected_real_vs_deranged,
            within_variable_and_origin_correlations=associations,within_variable_val_tail_groups=within))
        print('EFFECT SENSITIVITY',name,flush=True)
    result=dict(added_after_main_results_inspected=True,records=records,
        caveat='Uniform POST-HOC controls, no per-dataset winner selected; conditional correlations still are not causal. Projection uses only forecasts and correction; same correction norm control separates direction from mere shrinkage')
    write_json(root/'sensitivity_posthoc.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    run(Path(p.parse_args().root))
