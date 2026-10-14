"""Build adapter-independent five-case paths from already selected observed episodes."""
import argparse
import json
from pathlib import Path
import numpy as np
from .query_future_state_rag import FutureStatePredictor
from .future_state_encoder import history_features,predict,decode
from .safe_error_policy import bound_delta
from .benchmark_error_correction import load
from .common import sha256,write_json


def augment(p,trace,data,limits):
    result=dict(trace);outputs={k:np.zeros_like(data['base']) for k in ['direct_copy','direct_transport','direct_consensus','direct_encoder']}
    for sid in np.unique(data['sid']):
        sid=int(sid);rows=np.flatnonzero(data['sid']==sid);group=p.groups[str(sid)] if p.grouped else list(range(len(p.stds)));local=group.index(sid)
        if p.grouped:
            name=f'encoder_{sid}.npz'
            if sha256(p.folder/name)!=p.run['group_model_hashes'][name]:raise ValueError('Changed encoder')
            with np.load(p.folder/name,allow_pickle=False) as f:state={k:f[k].copy() for k in f.files}
        else:state=p.encoder
        width=int(state.get('block_width',16));hw=int(state.get('history_width',16));starts=data['start'][rows];selected=trace['selected_starts'][rows]
        if np.any(selected+192>np.minimum(starts[:,None],int(limits[str(sid)]))):raise ValueError('Unobserved/self/overlapping direct case')
        histories=np.stack([p.joint_window(int(t),96,group) for t in starts]);histories[:,:,local]=data['raw_x'][rows]
        codes=predict(state,history_features(histories,starts,p.means[group],p.stds[group],p.base_run['period'] or 1,weekly=p.grouped,block_width=hw))
        query_scale=np.maximum(data['raw_x'][rows].std(1),.1*p.stds[sid]);query_level=data['raw_x'][rows,-16:].mean(1)
        encoded=np.stack([decode(code,len(group),width)[:,local] for code in codes])*query_scale[:,None]+query_level[:,None]
        old=np.stack([p.joint_window(int(t),96,group) for t in selected.reshape(-1)])
        oldcodes=predict(state,history_features(old,selected.reshape(-1),p.means[group],p.stds[group],p.base_run['period'] or 1,weekly=p.grouped,block_width=hw))
        old_target=old[:,:,local].reshape(len(rows),5,96);old_scale=np.maximum(old_target.std(2),.1*p.stds[sid]);old_level=old_target[:,:,-16:].mean(2)
        old_encoded=np.stack([decode(code,len(group),width)[:,local] for code in oldcodes]).reshape(len(rows),5,96)*old_scale[:,:,None]
        actual=np.stack([p.synchronous_values[sid][int(t)+96:int(t)+192] for t in selected.reshape(-1)]).reshape(len(rows),5,96)
        ratio=np.clip(data['raw_x'][rows].std(1)[:,None]/old_scale,.25,4.)
        copied=query_level[:,None,None]+(actual-old_level[:,:,None])*ratio[:,:,None]
        innovation=(actual-old_level[:,:,None]-old_encoded)*ratio[:,:,None]
        mean=innovation.mean(1);variance=((innovation-mean[:,None])**2).mean(1);confidence=mean**2/(mean**2+variance+1e-12)
        paths=dict(direct_copy=copied.mean(1),direct_transport=encoded+mean,direct_consensus=encoded+confidence*mean,direct_encoder=encoded)
        for kind,path in paths.items():outputs[kind][rows]=bound_delta(path-data['base'][rows],data['raw_quantiles'][rows])/data['std'][rows,None]
    result.update({f'expert_{k}':v for k,v in outputs.items()});return result


def experiment(folder,source):
    run=json.loads((folder/'run.json').read_text());checkpoint=Path(run['source_checkpoint']);p=FutureStatePredictor(source/'store',checkpoint,adapter=False)
    limits={str(s['sid']):s['train_end'] if run.get('memory_scope')=='train' else s['memory_end'] for s in p.store.series};files={}
    try:
        for split in ['validation','test']:
            with np.load(folder/f'{split}_trace.npz',allow_pickle=False) as f:trace={k:f[k].copy() for k in f.files}
            data=load(p.cache,split);np.testing.assert_array_equal(trace['start'],data['start']);np.testing.assert_array_equal(trace['sid'],data['sid'])
            result=augment(p,trace,data,limits);path=folder/f'direct_{split}_trace.npz';np.savez_compressed(path,**result)
            files[split]=dict(source_trace_sha256=sha256(folder/f'{split}_trace.npz'),direct_trace_sha256=sha256(path));print('DIRECT CASE TRACE',folder.parent.name,folder.name,split,flush=True)
        write_json(folder/'direct_manifest.json',dict(source_run_sha256=sha256(folder/'run.json'),encoder_run_sha256=sha256(checkpoint/'run.json'),
            limits=limits,files=files,aggregation='Uniform mean of exactly the selected five cases',
            mechanism='Independent raw Moirai reference; no rejected base adapter output enters these paths'))
    finally:p.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008');p.add_argument('--datasets',nargs='+',required=True);a=p.parse_args()
    for name in a.datasets:experiment(Path(a.root)/name,Path(a.source)/name)
