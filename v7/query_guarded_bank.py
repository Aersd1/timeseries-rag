"""Deploy a validation-frozen choice of one five-case retrieval branch."""
import json
from pathlib import Path
import numpy as np
from .common import sha256
from .query_future_state_rag import FutureStatePredictor
from .conditional_recall import ConditionalPredictor
from .analogue_transport import experts,direct_experts


class GuardedBankPredictor:
    def __init__(self,store_path,folder,device='cuda',adapter=None):
        self.folder=Path(folder);self.store_path=Path(store_path);self.device=device
        run=json.loads((self.folder/'run.json').read_text());contract=json.loads((self.folder/'contract.json').read_text())
        if sha256(self.folder/'policy.json')!=run['policy_sha256'] or sha256(self.folder/'contract.json')!=run['contract_sha256']:raise ValueError('Changed bank policy/contract')
        self.policy=json.loads((self.folder/'policy.json').read_text());self.configs=contract['branch_runs']
        self.no_rag_policy=None
        if contract.get('direct'):
            if sha256(self.folder/'no_rag_policy.json')!=run['no_rag_policy_sha256']:raise ValueError('Changed no-RAG control policy')
            self.no_rag_policy=json.loads((self.folder/'no_rag_policy.json').read_text())
        original=Path(contract['original']);source=json.loads((original/'run.json').read_text())
        if sha256(original/'run.json')!=contract['original_run_sha256']:raise ValueError('Changed original transport run')
        self.original=FutureStatePredictor(store_path,source['source_checkpoint'],device,adapter)
        self.branches={original.parent.name:self.original}

    def _predictor(self,branch):
        if branch not in self.branches:
            cfg=self.configs[branch];path=Path(cfg['path'])
            if sha256(path/'run.json')!=cfg['run_sha256']:raise ValueError('Changed retrieval branch')
            run=json.loads((path/'run.json').read_text());self.branches[branch]=ConditionalPredictor(self.store_path,run['source_checkpoint'],
                run['memory_scope'],self.device,self.original.adapter,recall_mode=run.get('recall_mode','conditional'))
        return self.branches[branch]

    def close(self):
        for predictor in set(self.branches.values()):predictor.close()

    def predict(self,history,sid,start):
        x=np.asarray(history,dtype=np.float32)
        if x.shape!=(96,) or not np.isfinite(x).all():raise ValueError('Expected finite96 history')
        q=self.original.adapter.forecast(x[None],96).cpu().numpy();probe=self.original.adapter.forecast(x[None,:64],32).cpu().numpy()
        return self.predict_cached(x,sid,start,q,probe)

    def predict_cached(self,history,sid,start,q,probe,training=False,metadata=None,validation=False):
        old=self.original.predict_cached(history,sid,start,q,probe,metadata=metadata,return_components=True,training=training,validation=validation)
        policy=self.policy[str(int(sid))];selected=old;forecast=old['forecasts']['routed']
        if policy['branch']!='incumbent':
            predictor=self._predictor(policy['branch'])
            selected=old if predictor is self.original else predictor.predict_cached(history,sid,start,q,probe,return_components=True,training=training,validation=validation)
            c=selected['components'];independent=policy['family'].startswith('direct_')
            delta=(direct_experts(c) if independent else experts(c,selected['weights']))[policy['family']]
            base=q[0,4] if independent else c['final_base'];forecast=(base+policy['beta']*delta*c['std']).tolist()
        forecasts={**old['forecasts'],'guarded_rag':forecast}
        if getattr(self,'no_rag_policy',None) is not None:
            p=self.no_rag_policy[str(int(sid))];c=old['components'];value=old['forecasts']['encoder_only']
            if p['branch']!='incumbent':value=(q[0,4]+p['beta']*direct_experts(c)['direct_encoder']*c['std']).tolist()
            forecasts['guarded_no_rag']=value
        weights=[.2]*5 if policy['family'].startswith('direct_') else selected['weights']
        return dict(forecasts=forecasts,selected_starts=selected['selected_starts'],weights=weights,
            branch=policy['branch'],family=policy['family'],beta=policy['beta'],
            query_future_used=False,length=96,horizon=96,auxiliary_cases=5)
