"""Validation chooses one alignment/strength once, then confirmation only rejects."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import sha256,write_json
from .benchmark_state_recall import metric,interval


def choose_routing(data,predictions,incumbent):
    out={};base=data['raw_quantiles'][:,4];truth=data['raw_y']
    for sid in np.unique(data['sid']):
        rows=np.flatnonzero(data['sid']==sid);rows=rows[np.argsort(data['start'][rows])];blocks=np.array_split(rows,3)
        default=dict(family='incumbent',beta=0.,reason='No stable incremental advantage')
        if min(map(len,blocks))<3:out[str(int(sid))]=default;continue
        options=[]
        for family in ['wiener','legacy']:
            change=predictions[family]-base
            for beta in [.1,.25,.5,1.]:
                proposed=incumbent+beta*change;gains=[]
                for ids in blocks[:2]:
                    a=truth[ids]-incumbent[ids];b=truth[ids]-proposed[ids]
                    gain=np.mean(a*a,1)-np.mean(b*b,1)
                    if gain.mean()<=.01*np.mean(a*a) or np.mean(gain>0)<.75 or np.mean(abs(b))>=np.mean(abs(a)):break
                    gains.extend(gain.tolist())
                else:options.append((-float(np.mean(gains)),family,beta))
        if not options:out[str(int(sid))]=default;continue
        _,family,beta=min(options);ids=blocks[2]
        a=truth[ids]-incumbent[ids];b=truth[ids]-incumbent[ids]-beta*(predictions[family][ids]-base[ids])
        gain=np.mean(a*a,1)-np.mean(b*b,1)
        ok=gain.mean()>0 and np.mean(gain>0)>=.75 and np.mean(abs(b))<np.mean(abs(a))
        out[str(int(sid))]=dict(family=family if ok else 'incumbent',beta=beta if ok else 0.,
            proposed_family=family,proposed_beta=beta,final_gain=float(gain.mean()),
            reason='Selected alignment confirmed' if ok else 'One selected alignment rejected; preserve incumbent')
    return out


def apply_routing(policy,incumbent,predictions,base,sids):
    result=np.asarray(incumbent).copy()
    for sid in np.unique(sids):
        p=policy[str(int(sid))];ids=np.asarray(sids)==sid
        if p['family']!='incumbent':result[ids]=incumbent[ids]+p['beta']*(predictions[p['family']][ids]-base[ids])
    return result


def experiment(folder):
    meta=json.loads((folder/'preparation.json').read_text());source=Path(meta['source'])
    if meta.get('case_count')!=5 or meta.get('residual_reference'):raise ValueError('Joint policy requires a non-reference five-case model')
    with np.load(folder/'validation.npz',allow_pickle=False) as f:data={k:f[k].copy() for k in ['raw_y','raw_quantiles','sid','start']}
    with np.load(folder/'validation_forecasts.npz',allow_pickle=False) as f:pred={k:f[k].copy() for k in ['wiener','legacy']}
    with np.load(source/'validation_trace.npz',allow_pickle=False) as f:
        for k in ['sid','start']:np.testing.assert_array_equal(data[k],f[k])
        incumbent=f['incumbent'].copy()
    policy=choose_routing(data,pred,incumbent);write_json(folder/'routing_policy.json',policy)
    write_json(folder/'routing_contract.json',dict(model_run_sha256=sha256(folder/'run.json'),
        validation_sha256=sha256(folder/'validation.npz'),validation_predictions_sha256=sha256(folder/'validation_forecasts.npz'),
        original_validation_sha256=sha256(source/'validation_trace.npz'),policy_sha256=sha256(folder/'routing_policy.json'),
        rule='Choose alignment and strength on first two VAL blocks; last can only confirm or reject this single selection',
        caveat='Additional development after inspecting fixed-alignment results; same previously seen TEST periods, not blind'))
    with np.load(folder/'predictions.npz',allow_pickle=False) as f:
        truth={k:f[k].copy() for k in ['raw_y','std','sid','start']};incumbent=f['previous_previous_five'].copy()
        pred={k:f['raw_'+k].copy() for k in ['wiener','legacy']};base=f['moirai_direct'].copy()
        control=f['increment_without_retrieval'].copy()
    prediction=apply_routing(policy,incumbent,pred,base,truth['sid'])
    np.savez_compressed(folder/'routed_predictions.npz',prediction=prediction,**truth)
    record=dict(complete=True,queries=len(prediction),dataset=folder.name,metrics=metric(prediction,truth),
        matched_query_increment_metrics=metric(control,truth),active_variables=[int(s) for s,p in policy.items() if p['family']!='incumbent'],
        contract_sha256=sha256(folder/'routing_contract.json'),prediction_sha256=sha256(folder/'routed_predictions.npz'),
        vs_incumbent_interval=interval(incumbent,prediction,truth),vs_query_increment_interval=interval(control,prediction,truth),
        caveat='Two retrieval alignment choices versus one query-only control; additional search range; exploratory reused periods')
    write_json(folder/'routing_run.json',record);print('FIVE ROUTING',folder.name,record['metrics'],record['active_variables'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--datasets',nargs='+',required=True);a=p.parse_args()
    for name in a.datasets:experiment(Path(a.root)/name)
