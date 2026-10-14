"""Calibrate case source and use jointly; return exactly five auxiliary cases."""
import argparse
import json
from pathlib import Path
import numpy as np
from .guarded_bank_rag import choose_bank,apply_bank
from .benchmark_state_recall import metric,interval
from .common import sha256,write_json


def table(folder,split,direct=False):
    with np.load(folder/f"{'direct_' if direct else ''}{split}_trace.npz",allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def experiment(original,branches,out,direct=False):
    out.mkdir(parents=True,exist_ok=False);source=json.loads((original/'run.json').read_text());checkpoint=Path(source['source_checkpoint'])
    incumbent=json.loads((checkpoint/'policy.json').read_text());val=table(original,'validation',direct);tables={k:table(v,'validation',direct) for k,v in branches.items()}
    for branch,values in tables.items():
        for key in ['sid','start','raw_y']:np.testing.assert_array_equal(values[key],val[key])
        np.testing.assert_allclose(values['base'],val['base'],rtol=2e-5,atol=2e-4)
    policy=choose_bank(tables,val,incumbent['routed'],incumbent['encoder_only'],direct=direct);write_json(out/'policy.json',policy)
    if direct:
        no_val=dict(val,routed=val['encoder_only']);no_policy=choose_bank({original.parent.name:val},no_val,incumbent['encoder_only'],incumbent['encoder_only'],direct=True,no_retrieval=True)
        write_json(out/'no_rag_policy.json',no_policy)
    write_json(out/'contract.json',dict(original=str(original),original_run_sha256=sha256(original/'run.json'),
        branch_runs={k:dict(path=str(v),run_sha256=sha256(v/'run.json'),
            validation_sha256=sha256(v/f"{'direct_' if direct else ''}validation_trace.npz"),
            direct_manifest_sha256=sha256(v/'direct_manifest.json') if direct else None) for k,v in branches.items()},
        original_validation_sha256=sha256(original/f"{'direct_' if direct else ''}validation_trace.npz"),policy_sha256=sha256(out/'policy.json'),direct=direct,
        rule='First two validation time blocks choose one branch; last only confirms or rejects against original RAG; each prediction uses exactly one set of five cases',
        caveat='Exploratory reused periods; multiple branches evaluated, not preregistered blind research'))
    test=table(original,'test',direct);tables={k:table(v,'test',direct) for k,v in branches.items()}
    for branch,values in tables.items():
        np.testing.assert_array_equal(values['sid'],test['sid']);np.testing.assert_array_equal(values['start'],test['start'])
    pred,selected=apply_bank(tables,test,policy);truth={k:test[k] for k in ['raw_y','std','sid','start']}
    forecasts={k:test[k] for k in ['moirai_direct','no_rag','encoder_only','routed']};forecasts['guarded_rag']=pred
    if direct:forecasts['guarded_no_rag']=apply_bank({original.parent.name:test},dict(test,routed=test['encoder_only']),no_policy)[0]
    np.savez_compressed(out/'predictions.npz',**forecasts,**truth,selected_starts=selected)
    metrics={k:metric(v,truth) for k,v in forecasts.items()}
    result=dict(complete=True,dataset=original.name,queries=len(test['sid']),metrics=metrics,policy_sha256=sha256(out/'policy.json'),
        contract_sha256=sha256(out/'contract.json'),predictions_sha256=sha256(out/'predictions.npz'),
        adopted_variables=[int(k) for k,p in policy.items() if p['branch']!='incumbent'],
        gain_interval=interval(test['routed'],pred,truth),rag_interval=interval(forecasts.get('guarded_no_rag',test['encoder_only']),pred,truth),
        caveat='Same frozen encoder and foundation outputs; changing memory from prefix to frozen TRAIN is reported explicitly')
    if direct:result['no_rag_policy_sha256']=sha256(out/'no_rag_policy.json')
    write_json(out/'run.json',result);print('GUARDED BANK RESULT',original.name,metrics,result['adopted_variables'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--branches',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--direct',action='store_true');a=p.parse_args()
    branches={Path(v).name:Path(v) for v in a.branches}
    for name in a.datasets:experiment(Path(a.original)/name,{k:v/name for k,v in branches.items()},Path(a.output)/name,a.direct)
