"""Use frozen A/Moirai plus a validated query-reliability gate; history only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from v6.data import scale_floor
from .common import sha256,write_json
from .learned_rerank import LearnedMoiraiReranker
from .quality_gate import quality_features,gate_predict


def predict(gate,store,checkpoint,index,x,sid,start,device='cpu'):
    gate=Path(gate); state=torch.load(gate,map_location='cpu',weights_only=True)
    run=json.loads((gate.parent/'gate_run.json').read_text())
    if not run['complete'] or run['checkpoint_sha256']!=sha256(gate) or state['family']!='a100_reliability_gate':
        raise ValueError('Incomplete or changed reliability model')
    if state['checkpoint_sha256']!=sha256(checkpoint) or state['index_sha256']!=sha256(Path(index)/'manifest.json'):
        raise ValueError('Reliability model uses different A weights or index')
    engine=LearnedMoiraiReranker(store,checkpoint,index,device,candidates=100)
    try:
        if state['data_id']!=engine.store.meta['data_id'] or state['config']!=engine.c:
            raise ValueError('Reliability model/store mismatch')
        prepared=engine.prepare(x,sid,start); result=engine.rank(prepared,.2,state['diversity_weight'])
        if len(result['hits'])!=5: raise ValueError('Require full top-five auxiliary memory')
        lookup={h['start']:i for i,h in enumerate(prepared['candidates'])}; ids=[lookup[h['start']] for h in result['hits']]
        series=engine.store.series[sid]
        feature=quality_features(x,scale_floor(series,engine.c),prepared['quantiles'],prepared['quantile_levels'],
            prepared['mapped'][ids],result['hits'],engine.c['length']+max(engine.c['horizons']),series['memory_std'])
        raw,adopted=gate_predict(state,feature[None]); alpha=float(adopted[0])
        prediction=(1-alpha)*prepared['direct_prediction']+alpha*result['prediction']
        return dict(prediction=prediction.tolist(),moirai_prediction=prepared['direct_prediction'].tolist(),
            analog_prediction=result['prediction'].tolist(),hits=result['hits'],analogue_alpha=alpha,
            raw_gate_alpha=float(raw[0]),adopted=state['adopted'],candidates=100,top_k=5,
            history_weight=.2,future_weight=.8,diversity_weight=state['diversity_weight'])
    finally: engine.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['gate','store','checkpoint','index','history','output']: p.add_argument('--'+name,required=True)
    p.add_argument('--sid',type=int,required=True); p.add_argument('--start',type=int,required=True)
    p.add_argument('--device',default='cpu'); args=p.parse_args(); output=Path(args.output)
    if output.exists(): raise FileExistsError('Choose a fresh output')
    write_json(output,predict(args.gate,args.store,args.checkpoint,args.index,np.load(args.history,allow_pickle=False),args.sid,args.start,args.device))
