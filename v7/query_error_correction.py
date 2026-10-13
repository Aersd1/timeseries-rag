"""History-only inference: raw Moirai plus an external retrieved-error head."""
import argparse
import numpy as np
import torch
from v5.data import Store
from .moirai import MoiraiAdapter
from .trend_rag import ResidualPool
from .error_correction import retrieve,features,innovations,predict_delta
from .common import write_json


def predict(store_path,checkpoint,history,sid,start,device='cuda'):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['variant']!='retrieval_error_correction_96':raise ValueError('Wrong correction checkpoint')
    store=Store(store_path);x=np.asarray(history,dtype=np.float32)
    if x.shape!=(96,) or not np.isfinite(x).all():raise ValueError('Expected finite96 history')
    try:
        if store.meta['data_id']!=state['source_data_id']:raise ValueError('Store identity mismatch')
        if not 0<=sid<len(store.series) or start<store.series[sid]['memory_end']:raise ValueError('Query must follow historical memory')
        adapter=MoiraiAdapter.pretrained(dict(model_id=state['model_identity']['model_id'],
            revision=state['model_identity']['revision'],cache_dir='.moirai-cache',local_files_only=True),device)
        quantiles=adapter.forecast(x[None],96)[0].cpu().numpy()
        pool=ResidualPool(store,sid,state['window'],end=store.series[sid]['n'] if state.get('memory_scope')=='observed' else None,
            stride=state.get('memory_stride',1))
        q,memory=retrieve(x,quantiles,pool,start,state['window'])
        std=max(store.series[sid]['memory_std'],1e-6)
        if state.get('value_mode','errors')=='innovations':
            errors,_=innovations(x,quantiles,pool,memory,std,state['window'],start)
        else:
            past=np.stack([store.window(sid,int(p),96) for p in memory['selected_starts']])
            old_predictions=adapter.forecast(past,96)[:,4].cpu().numpy()
            old_truth=np.stack([store.window(sid,int(p)+96,96) for p in memory['selected_starts']])
            errors=((old_truth-old_predictions)/std).astype(np.float32)
        vector=features(x,quantiles,q,memory,errors,std)
        delta=predict_delta(state,dict(features=vector[None],keys=memory['examples'][None,:,:96],
            errors=errors[None],weights=memory['weights'][None]))[0]*std
        return dict(prediction=(quantiles[4]+delta).tolist(),moirai_prediction=quantiles[4].tolist(),
            correction=delta.tolist(),selected_starts=memory['selected_starts'].tolist(),
            weights=memory['weights'].tolist(),adopted=state['adopted'],beta=state['beta'],
            length=96,horizon=96,value_mode=state.get('value_mode','errors'),memory_scope=state.get('memory_scope','prefix'),
            mechanism='External correction head; original raw Moirai forecasting path stays frozen')
    finally:store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--store',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--history',required=True);p.add_argument('--sid',required=True,type=int);p.add_argument('--start',required=True,type=int)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');args=p.parse_args();torch.set_num_threads(4)
    write_json(args.output,predict(args.store,args.checkpoint,np.load(args.history,allow_pickle=False),args.sid,args.start,args.device))
