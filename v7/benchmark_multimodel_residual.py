"""Executable H1 matrix of the multi-model protocol; no TEST-selected strategy."""
import argparse,gc,time
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from .common import sha256,write_json
from .benchmark_memory_access import read
from .current_forecast import DATASETS
from .moirai import MoiraiAdapter,MODEL_ID,MODEL_REVISION
from .multimodel_residual import (L,H,PHASES,MODELS,ARMS,plan,materialize,
    train_neural,train_ridge,predict,neural_model,choose_cases,feature_blocks,fit_probe,apply_probe,norm,query_metadata)


def arrays(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def packet(name,folder,seeds):
    source=Path('v7/runs/all_datasets_20261008')/name/'store'
    if (folder/'packet.json').exists():
        old=read(folder/'packet.json')
        for file,h in old['files'].items():
            if sha256(folder/file)!=h:raise ValueError('Changed packet')
        return old
    folder.mkdir(parents=True,exist_ok=True);store=Store(source)
    try:
        p=plan(store);refs=p.pop('refs');files={}
        for phase in ('A','B','C','D','E'):
            d=materialize(store,refs[phase],p['stats']);path=folder/f'data_{phase}.npz'
            np.savez_compressed(path,**d);files[path.name]=sha256(path)
        path=folder/'test_refs.npz';np.savez_compressed(path,refs=refs['F']);files[path.name]=sha256(path)
        bank=arrays(folder/'data_C.npz')
        for phase in ('D','E'):
            q=arrays(folder/f'data_{phase}.npz')
            for seed in seeds:
                real,random=choose_cases(q,bank,seed);path=folder/f'cases_{phase}_{seed}.npz'
                np.savez_compressed(path,real=real,random=random);files[path.name]=sha256(path)
        source_config=read(source.parent/'config.json')
        record=dict(p,dataset=name,store=str(source),files=files,
            daily_period=86400/float(source_config['data']['sources'][0]['expected_interval']),
            phase_rows={k:len(v) for k,v in refs.items()},history=L,horizon=H)
        write_json(folder/'packet.json',record);print('MULTI PACKET',name,record['phase_rows'],flush=True);return record
    finally:store.close()


def model_path(folder,kind,seed):return folder/f'{kind}_{seed if kind in ("patchtst","dlinear") else "fixed"}'


def restore(path,kind,device,moirai=None):
    if kind=='moirai2':return moirai
    if kind=='ridge':return arrays(path/'backbone.npz')['coef']
    state=torch.load(path/'backbone.pt',map_location=device,weights_only=True)
    model=neural_model(kind).to(device);model.load_state_dict(state);return model.eval()


def backbone(folder,kind,seed,device,moirai,max_epochs):
    path=model_path(folder,kind,seed);path.mkdir(exist_ok=True)
    record_file=path/'backbone.json'
    if record_file.exists():
        record=read(record_file)
        for file,h in record['files'].items():
            if sha256(path/file)!=h:raise ValueError('Changed model cache')
        return path
    began=time.perf_counter();a=arrays(folder/'data_A.npz');b=arrays(folder/'data_B.npz');files={}
    if kind=='moirai2':model=moirai;record=dict(model_id=MODEL_ID,revision=MODEL_REVISION,training=False)
    elif kind=='ridge':
        model,record=train_ridge(a,b);np.savez_compressed(path/'backbone.npz',coef=model);files['backbone.npz']=sha256(path/'backbone.npz')
    else:
        model,record=train_neural(kind,a,b,seed,device,max_epochs)
        torch.save(model.cpu().state_dict(),path/'backbone.pt');files['backbone.pt']=sha256(path/'backbone.pt');model.to(device)
    del a,b;gc.collect()
    saved={}
    for phase in ('C','D','E'):
        d=arrays(folder/f'data_{phase}.npz');saved[phase]=predict(model,kind,d,device);del d
    np.savez_compressed(path/'base_predictions.npz',**saved);files['base_predictions.npz']=sha256(path/'base_predictions.npz')
    write_json(record_file,dict(record,kind=kind,seed=seed,history=L,horizon=H,files=files,seconds=time.perf_counter()-began))
    print('MULTI BACKBONE',folder.name,kind,seed,round(time.perf_counter()-began,1),flush=True)
    if kind not in ('moirai2','ridge'):del model;torch.cuda.empty_cache()
    return path


def losses(pred,y,sd):
    error=np.asarray(pred,dtype=float)-y
    return np.stack([(error**2).mean(1),abs(error).mean(1),((error/sd[:,None])**2).mean(1)],1)


def develop(folder,kind,seed,path):
    output=path/f'development_{seed}.json'
    if output.exists():
        record=read(output)
        for file,h in record['files'].items():
            if sha256(path/file)!=h:raise ValueError('Changed probe')
        return
    packet_info=read(folder/'packet.json')
    d=query_metadata(arrays(folder/'data_D.npz'),packet_info);e=query_metadata(arrays(folder/'data_E.npz'),packet_info);bank=arrays(folder/'data_C.npz')
    bases=arrays(path/'base_predictions.npz');di=arrays(folder/f'cases_D_{seed}.npz');ei=arrays(folder/f'cases_E_{seed}.npz')
    _,_,scale=norm(d['x'],d['std']);target=(d['y']-bases['D'])/scale[:,None]
    result={};files={}
    for arm in ARMS:
        common,extra,scale=feature_blocks(d,bases['D'],bank,bases['C'],di['real'],di['random'],arm,seed)
        probe=fit_probe(common,extra,target,seed);file=f'probe_{seed}_{arm}.npz';np.savez_compressed(path/file,**probe);files[file]=sha256(path/file)
        ec,ex,es=feature_blocks(e,bases['E'],bank,bases['C'],ei['real'],ei['random'],arm,seed)
        pred=apply_probe(probe,ec,ex,bases['E'],es);v=losses(pred,e['y'],e['std'])
        result[arm]=dict(mse=float(v[:,0].mean()),mae=float(v[:,1].mean()),train_nmse=float(v[:,2].mean()),effective_df=float(probe['effective_df']))
    write_json(output,dict(kind=kind,seed=seed,metrics=result,files=files,probe_training='D only; E diagnostic, no TEST selection'))
    print('MULTI PROBES',folder.name,kind,seed,flush=True)


def evaluate(folder,kind,seed,path,device,moirai,frozen):
    output=path/f'test_{seed}.json'
    if output.exists():
        result=read(output)
        if sha256(path/f'test_rows_{seed}.npz')!=result['rows_sha256']:raise ValueError('Changed TEST results')
        return result
    packet=read(folder/'packet.json');store=Store(packet['store']);refs=arrays(folder/'test_refs.npz')['refs']
    try:
        if sha256(store.path/'catalog.json')!=packet['catalog_sha256']:raise ValueError('Changed source catalog')
        for sid,h in packet['values_sha256'].items():
            if sha256(store.path/store.series[int(sid)]['file'])!=h:raise ValueError('Changed physical source')
        d=query_metadata(materialize(store,refs,packet['stats'],targets=False),packet)
        bank=arrays(folder/'data_C.npz');cases_file=folder/f'cases_F_{seed}.npz'
        if cases_file.exists():ids=arrays(cases_file);real,random=ids['real'],ids['random']
        else:
            real,random=choose_cases(d,bank,seed);np.savez_compressed(cases_file,real=real,random=random)
        model=restore(path,kind,device,moirai);base=predict(model,kind,d,device);case_base=arrays(path/'base_predictions.npz')['C']
        forecasts={'base':base};dfs=[]
        for arm in ARMS:
            file=f'probe_{seed}_{arm}.npz'
            if sha256(path/file)!=frozen[str(path/file)]:raise ValueError('Probe changed after freeze')
            common,extra,scale=feature_blocks(d,base,bank,case_base,real,random,arm,seed)
            probe=arrays(path/file);dfs.append(float(probe['effective_df']))
            forecasts[arm]=apply_probe(probe,common,extra,base,scale)
        if max(dfs)-min(dfs)>.05:raise ValueError('Effective capacity not matched')
        # No F label was needed by retrieval, base forecast or residual prediction.
        y=np.stack([store.window(s,t+L,H) for s,t in refs]);scores={};values={}
        for arm,pred in forecasts.items():
            v=losses(pred,y,d['std']);values['loss_'+arm]=v
            scores[arm]=dict(zip(('mse','mae','train_nmse'),v.mean(0).tolist()))
        np.savez_compressed(path/f'test_rows_{seed}.npz',sid=d['sid'],start=d['start'],
            raw_y=y,base=base,query_only=forecasts['query_only'],real=forecasts['real'],**values)
        result=dict(dataset=folder.name,model=kind,seed=seed,queries=len(refs),variables=len(packet['sids']),
            origins=len(np.unique(d['start'])),metrics=scores,effective_df_range=[min(dfs),max(dfs)],
            rows_sha256=sha256(path/f'test_rows_{seed}.npz'))
        write_json(output,result);print('MULTI RESULT',folder.name,kind,seed,
            'extra_nmse',round(100*(1-scores['real']['train_nmse']/scores['query_only']['train_nmse']),3),flush=True)
        if kind not in ('moirai2','ridge'):del model;torch.cuda.empty_cache()
        return result
    finally:store.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS));p.add_argument('--seeds',nargs='+',type=int,default=[13,41,73])
    p.add_argument('--device',default='cuda');p.add_argument('--max-epochs',type=int,default=32);a=p.parse_args()
    root=Path(a.output);codes=['v7/multimodel_residual.py','v7/benchmark_multimodel_residual.py','v7/MULTI_MODEL_RAG_PROTOCOL.md']
    protocol=dict(datasets=a.datasets,models=list(MODELS),seeds=a.seeds,history=L,horizon=H,max_epochs=a.max_epochs,
        query_state='Training-relative level, log amplitude, daily and weekly phase, elapsed-time fraction in every arm',
        time_grid='All phases anchored to global multiples of their stride; common forecast clock across memory/D/E/F',
        phases={k:list(v) for k,v in PHASES.items()},arms=list(ARMS),code_sha256={f:sha256(f) for f in codes},
        primary='Real residual probe versus query-only, matched-random and averaged deranged losses',
        independent_memory=True,capacity='128 fixed random features; matched effective_df=min64,N_D/2',
        status='Predeclared exploratory H1 matrix, not completion of H2-H5 or four-level gate validation',
        caveat='Previously explored datasets; not a fresh external blind test; local model implementations and finite training budget')
    if root.exists():
        if not a.resume or read(root/'protocol.json')!=protocol:raise ValueError('Use exact matching --resume or fresh output')
    else:root.mkdir(parents=True);write_json(root/'protocol.json',protocol)
    torch.set_num_threads(4);began=time.perf_counter()
    for name in a.datasets:packet(name,root/name,a.seeds)
    print('MULTI MOIRAI LOADING',flush=True)
    moirai=MoiraiAdapter.pretrained(dict(model_id=MODEL_ID,revision=MODEL_REVISION,cache_dir='.moirai-cache',local_files_only=True),a.device)
    for name in a.datasets:
        folder=root/name
        for kind in MODELS:
            for seed in a.seeds:
                path=backbone(folder,kind,seed,a.device,moirai,a.max_epochs)
                develop(folder,kind,seed,path);gc.collect()
    frozen={}
    for name in a.datasets:
        for file in (root/name).rglob('*'):
            if file.is_file() and not file.name.startswith(('test_','cases_F_')):frozen[str(file)]=sha256(file)
    if (root/'frozen.json').exists():
        if read(root/'frozen.json')!=frozen:raise ValueError('Changed development freeze')
    else:write_json(root/'frozen.json',frozen)
    for file,h in protocol['code_sha256'].items():
        if sha256(file)!=h:raise ValueError('Code changed during experiment')
    records=[]
    for name in a.datasets:
        for kind in MODELS:
            for seed in a.seeds:
                path=model_path(root/name,kind,seed)
                records.append(evaluate(root/name,kind,seed,path,a.device,moirai,frozen));gc.collect()
    write_json(root/'summary.json',dict(complete=True,seconds=time.perf_counter()-began,protocol=protocol,records=records))


if __name__=='__main__':main()
