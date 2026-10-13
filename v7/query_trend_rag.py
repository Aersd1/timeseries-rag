"""History-only 96-to-96 inference with internal residual retrieval attention."""
import argparse
import numpy as np
import torch
from v5.data import Store
from .common import write_json
from .moirai import MoiraiAdapter
from .trend_rag import decompose, normalize, ResidualPool
from .retrieval_finetune import RetrievalFineTune


def predict(store_path,checkpoint,history,sid,start,device='cuda'):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['variant']!='trend_residual_rag_96' or not state['use_retrieval']:
        raise ValueError('Expected residual RAG checkpoint')
    x=np.asarray(history,dtype=np.float32)
    if x.shape!=(96,) or not np.isfinite(x).all(): raise ValueError('Expected finite96 history')
    store=Store(store_path)
    try:
        if store.meta['data_id']!=state['source_data_id']: raise ValueError('Wrong retrieval store')
        if not 0<=sid<len(store.series) or start<store.series[sid]['memory_end']:
            raise ValueError('Query must follow fixed memory')
        settings=dict(model_id=state['identity']['model_id'],revision=state['identity']['revision'],
            cache_dir='.moirai-cache',local_files_only=True)
        adapter=MoiraiAdapter.pretrained(settings,device)
        median=list(adapter.module.quantile_levels).index(.5)
        trend,residual=decompose(x,state['window']);pool=ResidualPool(store,sid,state['window'])
        normalized,mean,scale=normalize(residual,pool.floor)
        trend_prediction=adapter.forecast(trend[None],96)[0,median].cpu().numpy()
        frozen_quantiles=adapter.forecast(normalized[None],96)[0].cpu().numpy()
        memory=pool.retrieve(normalized,start,frozen_quantiles,adapter.module.quantile_levels)
        model=RetrievalFineTune(adapter,96,96,use_retrieval=True,fusion='paired',rank=state['rank']).to(device)
        model.load_adaptation(state['adaptation']);model.eval()
        with torch.no_grad():
            q=model(torch.tensor(normalized[None],device=device),torch.tensor(memory['examples'][None],device=device),
                torch.tensor(memory['weights'][None],device=device),torch.tensor(memory['valid'][None],device=device))
        residual_prediction=q[0,median].cpu().numpy()*float(scale[0])+float(mean[0])
        return dict(prediction=(trend_prediction+residual_prediction).tolist(),trend_prediction=trend_prediction.tolist(),
            residual_prediction=residual_prediction.tolist(),selected_starts=memory['selected_starts'].tolist(),
            weights=memory['weights'].tolist(),gate_mean=float(model.retrieval.last_gate.mean()),
            length=96,horizon=96,window=state['window'],checkpoint_epoch=state['epoch'],
            conditioning='Internal Moirai attention to old residual-history/continuation pairs')
    finally: store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--store',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--history',required=True);p.add_argument('--sid',type=int,required=True);p.add_argument('--start',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');args=p.parse_args()
    torch.set_num_threads(4)
    write_json(args.output,predict(args.store,args.checkpoint,np.load(args.history,allow_pickle=False),args.sid,args.start,args.device))
