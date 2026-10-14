"""Deploy the frozen 5+5 policy from actual history, without query outcomes."""
import json
from pathlib import Path
import numpy as np
from .common import sha256
from .query_guarded_bank import GuardedBankPredictor
from .dual_five_rag import RawHistoryBank, encoded_cases, fusion, incumbent_reference


class DualFivePredictor:
    def __init__(self, store_path, folder, device='cuda', adapter=None):
        self.folder = Path(folder)
        run = json.loads((self.folder/'run.json').read_text())
        self.contract = json.loads((self.folder/'contract.json').read_text())
        if sha256(self.folder/'policy.json')!=run['policy_sha256'] or sha256(self.folder/'contract.json')!=run['contract_sha256']:
            raise ValueError('Changed dual-five policy/contract')
        source = Path(self.contract['source'])
        if sha256(source/'run.json')!=self.contract['source_run_sha256'] or sha256(source/'contract.json')!=self.contract['source_contract_sha256']:
            raise ValueError('Changed original five-case source')
        self.policy = json.loads((self.folder/'policy.json').read_text())
        self.previous = GuardedBankPredictor(store_path, source, device, adapter)
        self.banks = {}

    def close(self): self.previous.close()

    def predict(self, history, sid, start):
        x = np.asarray(history, dtype=np.float32)
        if x.shape!=(96,) or not np.isfinite(x).all(): raise ValueError('Expected finite96 history')
        adapter = self.previous.original.adapter
        q = adapter.forecast(x[None], 96).cpu().numpy()
        probe = adapter.forecast(x[None, :64], 32).cpu().numpy()
        return self.predict_cached(x, sid, start, q, probe)

    def predict_cached(self, history, sid, start, q, probe):
        sid, start = int(sid), int(start)
        x = np.asarray(history, dtype=np.float32)
        previous = self.previous.predict_cached(x, sid, start, q, probe)
        p = self.previous.original
        end = int(self.contract['limits'][str(sid)])
        if sid not in self.banks:
            if len(self.banks)>=4: self.banks.pop(next(iter(self.banks)))
            self.banks[sid] = RawHistoryBank(p.store, sid, end, self.contract['memory_stride'], self.contract.get('raw_case_spacing',0))
        raw, distance = self.banks[sid].select(x, start, previous['selected_starts'])
        selected = np.r_[previous['selected_starts'], raw].astype(np.int64)
        if len(np.unique(selected))!=10 or np.any(selected+192>min(start, end)):
            raise ValueError('Invalid ten-case temporal boundary or duplicate')
        encoded, copied, innovation = encoded_cases(p, sid, x[None], np.array([start]), selected[None])
        incumbent = np.asarray(previous['forecasts']['guarded_rag'])[None]
        paths, state = fusion(encoded, copied, innovation, q, x[None], np.array([p.stds[sid]]), distance[None], incumbent)
        policy = self.policy[str(sid)]
        output = incumbent[0]
        if policy['family']!='incumbent':
            base = incumbent[0] if incumbent_reference(policy['family']) else np.asarray(q)[0, 4]
            output = base+policy['beta']*(paths[policy['family']][0]-base)
        forecasts = {**previous['forecasts'], 'guarded_dual':output.tolist()}
        for family, value in paths.items():
            base = incumbent[0] if incumbent_reference(family) else np.asarray(q)[0, 4]
            beta = 1. if family=='dual_trust' else .25
            forecasts['fixed_'+family] = (base+beta*(value[0]-base)).tolist()
        old_policy = self.previous.policy[str(sid)]
        if old_policy['branch']=='incumbent':
            old_policy = p.policies['routed'][str(sid)]
        old_uses_cases = old_policy['beta']>0 and old_policy['family'] not in ['direct','encoder_only','direct_encoder']
        return dict(forecasts=forecasts, selected_starts=selected.tolist(),
            routed_starts=selected[:5].tolist(), raw_history_starts=selected[5:].tolist(),
            raw_history_distances=distance.tolist(), support=float(state['support'][0]),
            agreement=float(state['agreement'][0]), family=policy['family'], beta=policy['beta'],
            length=96, horizon=96, retrieved_cases=10,
            used_cases=(5 if old_uses_cases else 0) if policy['family']=='incumbent' else 10, query_future_used=False)
