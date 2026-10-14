"""Prepare causal TRAIN and frozen VAL/TEST 5+5 episodes for neural fusion."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import sha256,write_json
from .query_guarded_bank import GuardedBankPredictor
from .benchmark_error_correction import load
from .future_state_encoder import history_features,prototype_state,predict
from .benchmark_future_state_rag import retrieve_forecast
from .trend_rag import ResidualPool
from .dual_five_rag import RawHistoryBank


def encoder(p,sid):
    if not p.grouped: return p.encoder
    if sid not in p.encoders:
        name=f'encoder_{sid}.npz'
        if sha256(p.folder/name)!=p.run['group_model_hashes'][name]: raise ValueError('Changed encoder')
        with np.load(p.folder/name,allow_pickle=False) as f: p.encoders[sid]={k:f[k].copy() for k in f.files}
    return p.encoders[sid]


def select_training_cases(previous,data,row,predictor,state,group):
    sid=int(data['sid'][row]); start=int(data['start'][row]); x=data['raw_x'][row]
    if predictor is previous.original:
        m={k:data[k][row] for k in ['candidate_starts','history_scores','future_scores']}
    else:
        m=predictor.candidate_metadata(x,sid,start,data['raw_quantiles'][row],state,group)
    candidates=np.asarray(m['candidate_starts'])
    if np.any(candidates+192>min(start,predictor.memory_limit(predictor.store.series[sid]))): raise ValueError('Training retrieval overlaps its query')
    joint=predictor.joint_window(start,96,group); joint[:,group.index(sid)]=x
    code=predict(state,history_features(joint[None],np.array([start]),predictor.means[group],predictor.stds[group],
        predictor.base_run['period'] or 1,weekly=predictor.grouped,block_width=int(state.get('history_width',16))))[0]
    order=np.argsort(candidates); library=candidates[order]
    episodes=np.stack([predictor.joint_window(int(t),192,group) for t in library])
    prototypes=prototype_state(episodes[:,:96],episodes[:,96:],predictor.means[group],predictor.stds[group],int(state.get('block_width',16)))
    weight=predictor.selection['groups'][str(sid)]['retrieval_weight'] if predictor.grouped else predictor.selection['retrieval_weight']
    fields=dict(raw_x=x[None],sid=np.array([sid]),start=np.array([start]),metric_std=np.array([predictor.stds[sid]]),
        **{k:np.asarray(m[k])[None] for k in ['candidate_starts','history_scores','future_scores']})
    return retrieve_forecast(fields,0,data['base'][row],code,state['precision'],None,prototypes,
        episodes[:,:96],episodes[:,96:],library,weight,channel_index=group.index(sid))[1]


def prepare(source,data_source,out,split,previous=None):
    if split=='test' and (out/'preparation.json').exists():
        meta=json.loads((out/'preparation.json').read_text())
        if meta.get('case_count')==5:
            if not (out/'policy.json').exists():raise ValueError('Freeze five-case policy before TEST')
            return copy_subset(Path(meta['feature_source']),out,split)
    own=previous is None
    dual_contract=json.loads((source/'contract.json').read_text()); fixed=Path(dual_contract['source'])
    if own: previous=GuardedBankPredictor(data_source/'store',fixed,adapter=False)
    p=previous.original
    try:
        data=load(p.cache,split); n=len(data['sid'])
        selected=np.zeros((n,10),dtype=np.int64); past=np.zeros((n,10,96),dtype=np.float32); future=np.zeros_like(past)
        if split!='train':
            if split=='test' and not (out/'policy.json').exists(): raise ValueError('Freeze fusion policy before reading TEST')
            path=source/f'{split}_trace.npz'
            expected=(json.loads((source/'run.json').read_text())['test_trace_sha256'] if split=='test' else dual_contract['source_validation_sha256'])
            if sha256(path)!=expected: raise ValueError('Changed frozen selected cases')
            with np.load(path,allow_pickle=False) as f:
                for k in ['sid','start']:np.testing.assert_array_equal(f[k],data[k])
                selected=f['selected_starts'].copy()
        for sid in np.unique(data['sid']):
            sid=int(sid); rows=np.flatnonzero(data['sid']==sid); series=p.store.series[sid]
            end=int(dual_contract['limits'][str(sid)])
            bank=RawHistoryBank(p.store,sid,end,dual_contract['memory_stride'],dual_contract['raw_case_spacing']) if split=='train' else None
            choice=previous.policy[str(sid)]
            pred=p if choice['branch']=='incumbent' else previous._predictor(choice['branch'])
            state=encoder(p,sid); group=p.groups[str(sid)] if p.grouped else list(range(len(p.stds)))
            for row in rows:
                start=int(data['start'][row])
                if split=='train':
                    first=select_training_cases(previous,data,row,pred,state,group)
                    raw,_=bank.select(data['raw_x'][row],start,first); selected[row]=np.r_[first,raw]
                if len(np.unique(selected[row]))!=10 or np.any(selected[row]+192>min(start,end)):
                    raise ValueError('Self, duplicate, overlapping or unobserved case')
                for j,t in enumerate(selected[row]):
                    past[row,j]=p.synchronous_values[sid][int(t):int(t)+96]
                    future[row,j]=p.synchronous_values[sid][int(t)+96:int(t)+192]
            if sid%100==0:print('FUSION PREPARE',source.name,split,sid,n,flush=True)
        if split=='train':
            for s in p.store.series:
                rows=data['sid']==s['sid']
                if np.any(data['start'][rows]+192>s['train_end']):raise ValueError('Training query label beyond TRAIN')
        record={k:data[k] for k in ['raw_x','raw_y','raw_quantiles','sid','start','std']}
        record.update(selected_starts=selected,case_past=past,case_future=future)
        path=out/f'{split}.npz';np.savez_compressed(path,**record)
        info=dict(queries=n,sha256=sha256(path),case_pairs=n*10)
        if split=='train':
            means=[];stds=[]
            for s in p.store.series:
                values=p.synchronous_values[s['sid']][:s['train_end']]; values=values[np.isfinite(values)]
                means.append(float(values.mean()));stds.append(max(float(values.std()),1e-6))
            write_json(out/'stats.json',dict(mean=means,std=stds,train_end=[s['train_end'] for s in p.store.series]))
        print('FUSION PREPARED',source.name,split,info,flush=True)
        return info
    finally:
        if own:previous.close()


def initial(source,data_source,out):
    out.mkdir(parents=True,exist_ok=False)
    contract=json.loads((source/'contract.json').read_text()); previous=GuardedBankPredictor(data_source/'store',contract['source'],adapter=False)
    try:
        files={split:prepare(source,data_source,out,split,previous) for split in ['train','validation']}
        write_json(out/'preparation.json',dict(source=str(source),source_run_sha256=sha256(source/'run.json'),
            source_contract_sha256=sha256(source/'contract.json'),source_cache=str(previous.original.cache),
            cache_manifest_sha256=sha256(previous.original.cache/'manifest.json'),stats_sha256=sha256(out/'stats.json'),
            store=str(data_source/'store'),source_data_id=previous.original.store.meta['data_id'],
            period=previous.original.base_run['period'],files=files,limits=contract['limits'],
            training='Frozen existing retriever; all old192 end <= TRAIN query history START; query future is supervision only'))
    finally:previous.close()


def copy_subset(source,out,split):
    if split=='test':
        expected=json.loads((source/'run.json').read_text())['test_info']['sha256']
    else:expected=json.loads((source/'preparation.json').read_text())['files'][split]['sha256']
    if sha256(source/f'{split}.npz')!=expected:raise ValueError('Changed original feature source')
    with np.load(source/f'{split}.npz',allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    for key in ['selected_starts','case_past','case_future']:data[key]=data[key][:,:5].copy()
    path=out/f'{split}.npz';np.savez_compressed(path,**data)
    return dict(queries=len(data['sid']),case_pairs=len(data['sid'])*5,sha256=sha256(path))


def initial_five(source,out):
    out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((source/'preparation.json').read_text());stats=json.loads((source/'stats.json').read_text())
    stats['case_count']=5;write_json(out/'stats.json',stats)
    meta.update(case_count=5,feature_source=str(source),feature_source_preparation_sha256=sha256(source/'preparation.json'),
        feature_source_run_sha256=sha256(source/'run.json'),stats_sha256=sha256(out/'stats.json'),
        files={split:copy_subset(source,out,split) for split in ['train','validation']},
        training='Same frozen routed FIRST FIVE only; no raw-history extra five; complete old192 ends before query history start')
    write_json(out/'preparation.json',meta);print('PREPARED FIVE ONLY',out.name,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--data',default='v7/runs/all_datasets_20261008');parser.add_argument('--datasets',nargs='+',required=True)
    parser.add_argument('--five-only',action='store_true');args=parser.parse_args()
    for name in args.datasets:
        if args.five_only:initial_five(Path(args.source)/name,Path(args.output)/name)
        else:initial(Path(args.source)/name,Path(args.data)/name,Path(args.output)/name)
