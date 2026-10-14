"""Deploy frozen component policies through the history-only foundation API."""
import json
from pathlib import Path
import numpy as np
from .common import sha256
from .query_future_state_rag import FutureStatePredictor
from .component_rag import component_experts
from .benchmark_component_rag import forecasts


class ComponentPredictor(FutureStatePredictor):
    def __init__(self,store_path,folder,device='cuda',adapter=None):
        folder=Path(folder);run=json.loads((folder/'run.json').read_text(encoding='utf-8'))
        if sha256(folder/'policy.json')!=run['policy_sha256'] or sha256(folder/'contract.json')!=run['contract_sha256']:
            raise ValueError('Changed component policy or contract')
        contract=json.loads((folder/'contract.json').read_text(encoding='utf-8'));source=Path(run['source_checkpoint'])
        if sha256(source/'run.json')!=contract['source_run_sha256']:raise ValueError('Changed underlying encoder run')
        self.component_policies=json.loads((folder/'policy.json').read_text(encoding='utf-8'))
        super().__init__(store_path,source,device,adapter)

    def predict_cached(self,history,sid,start,q,probe,metadata=None,return_components=False,validation=False):
        if validation:raise ValueError('Deployment cannot search policies on calibration outcomes')
        result=super().predict_cached(history,sid,start,q,probe,metadata,True)
        c=result['components'];experts,confidence,effective=component_experts(c,result['weights'],result['selected_starts'])
        trace=dict(final_base=c['final_base'][None],std=np.array([c['std']]),sid=np.array([sid]),
            **{k:np.array(v)[None] for k,v in result['forecasts'].items()},
            **{f'expert_{k}':v[None] for k,v in experts.items()})
        result['forecasts']={k:v[0].tolist() for k,v in forecasts(trace,self.component_policies).items()}
        result.update(component_confidence=confidence.tolist(),effective_cases=effective,
            component_policy='validation-frozen, six16-point blocks, half shrinkage')
        if not return_components:result.pop('components')
        return result
