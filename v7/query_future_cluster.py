"""History-only scratch-encoder retrieval with known historical future-regime keys."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from v5.data import Store
from v6.data import scale_floor
from .common import sha256,write_json
from .future_cluster import FutureClusterEncoder,eligible,rank_from_history


def predict(folder,store_path,x,sid,start,mode='history_future_cluster',device='cpu'):
    folder=Path(folder); store=Store(store_path)
    try:
        meta=json.loads((folder/'index_manifest.json').read_text())
        if not meta['complete'] or meta['data_id']!=store.meta['data_id'] or meta.get('feature_policy')!='history_mean_scaled_ordered_future_v1':
            raise ValueError('Store/index identity differs')
        if sha256(folder/'index.npz')!=meta['index_sha256']:
            raise ValueError('Index changed')
        variant='scratch' if mode=='scratch_embedding' else 'cluster'
        checkpoint=folder/variant/'best.pt'
        run=json.loads((folder/variant/'run.json').read_text())
        if not run['complete'] or sha256(checkpoint)!=meta['checkpoints'][variant] or sha256(checkpoint)!=run['checkpoint_sha256']:
            raise ValueError('Checkpoint changed or incomplete')
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['family']!='future_cluster_scratch' or state['data_id']!=meta['data_id'] or state['source_config']!=meta['source_config'] or state.get('feature_policy')!=meta.get('feature_policy'):
            raise ValueError('Encoder/index mismatch')
        if mode not in ['scratch_embedding','cluster_embedding','future_cluster','history_future_cluster','history_future_regression']:
            raise ValueError('Unknown retrieval mode')
        c=meta['source_config']; m,h=c['length'],max(c['horizons'])
        x=np.asarray(x,np.float32)
        if x.shape!=(m,) or not np.isfinite(x).all() or not 0<=sid<len(store.series) or start<0:
            raise ValueError('Invalid history or query identity')
        model=FutureClusterEncoder(state['model_config'],state['use_clusters']).to(device)
        model.load_state_dict(state['model']); model.eval()
        floor=scale_floor(store.series[sid],c)
        with torch.no_grad(): out=model(torch.as_tensor(x[None],device=device),torch.tensor([floor],device=device))
        center,scale=float(x.mean()),max(float(x.std()),floor)
        query=dict(z=out['z'][0].cpu().numpy(),probability=None if out['probability'] is None else out['probability'][0].cpu().numpy(),
            normalized_history=(x-center)/scale,center=center,scale=scale,
            normalized_forecast=(out['forecast'][0].cpu().numpy()-center)/scale,horizons=c['horizons'])
        with np.load(folder/'index.npz',allow_pickle=False) as data:
            ok=eligible([sid],[start],data['sid'],data['start'],m+h)[0]
            ids=np.flatnonzero(ok)
            if len(ids)<5: raise ValueError('Need at least five complete nonoverlapping historical candidates')
            result=rank_from_history(query,data['past'][ok],data['future'][ok],data[variant+'_z'][ok],data['probability'][ok],
                mode if mode not in ['scratch_embedding','cluster_embedding'] else 'embedding')
            selected=ids[result['indices']]
            hits=[dict(sid=sid,start=int(data['start'][j]),end=int(data['start'][j])+m+h,weight=float(w))
                for j,w in zip(selected,result['weights'])]
        return dict(prediction=result['prediction'].tolist(),hits=hits,mode=mode,
            cluster_probability=None if query['probability'] is None else query['probability'].tolist(),
            future_weight=.8 if mode not in ['scratch_embedding','cluster_embedding'] else None,
            history_weight=.2 if mode not in ['scratch_embedding','cluster_embedding'] else None,
            model_family='future_cluster_scratch',no_pretrained_encoder=True)
    finally: store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['folder','store','history','output']: p.add_argument('--'+name,required=True)
    p.add_argument('--sid',type=int,required=True); p.add_argument('--start',type=int,required=True)
    p.add_argument('--mode',choices=['scratch_embedding','cluster_embedding','future_cluster','history_future_cluster','history_future_regression'],default='history_future_cluster')
    p.add_argument('--device',default='cpu'); args=p.parse_args()
    output=Path(args.output)
    if output.exists(): raise FileExistsError('Choose a fresh output')
    write_json(output,predict(args.folder,args.store,np.load(args.history,allow_pickle=False),args.sid,args.start,args.mode,args.device))
