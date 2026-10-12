import inspect
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from v7.quality_gate import FEATURES,ReliabilityGate,quality_features,forecast_loss,gate_predict
from v7.benchmark_a100 import select_policy,STRENGTHS,ALPHAS


class QualityGateTests(unittest.TestCase):
    def test_quality_features_use_history_and_known_candidates_only(self):
        x=np.sin(np.arange(40)/5); q=np.stack([np.zeros(16)-1,np.zeros(16),np.zeros(16)+1])
        mapped=np.stack([np.ones(16)*i*.1 for i in range(5)])
        hits=[dict(weight=.2,start=i*2,sid=0,history_nmse=.1,future_score=.2,distance=.03) for i in range(5)]
        f=quality_features(x,.1,q,[.1,.5,.9],mapped,hits,56,1.)
        self.assertEqual(f.shape,(len(FEATURES),)); self.assertTrue(np.isfinite(f).all())
        changed=quality_features(x,.1,q,[.1,.5,.9],mapped*10,hits,56,1.)
        self.assertGreater(changed[2],f[2]); self.assertGreater(changed[4],f[4])
        self.assertNotIn('y',inspect.signature(quality_features).parameters)
        self.assertNotIn('target',inspect.signature(quality_features).parameters)

    def test_gate_learns_query_dependent_reliability_and_reloads(self):
        torch.manual_seed(7); n=80
        features=torch.randn(n,len(FEATURES)); truth=(features[:,0]>0).float()
        items=dict(direct=torch.zeros(n,8),analog=torch.ones(n,8),target=truth[:,None].expand(-1,8),std=torch.ones(n))
        model=ReliabilityGate(.5); opt=torch.optim.Adam(model.parameters(),lr=.03)
        initial=float(forecast_loss(model(features),items,[4,8]).detach())
        for _ in range(100):
            opt.zero_grad(); loss=forecast_loss(model(features),items,[4,8]); loss.backward(); opt.step()
        model.eval()
        self.assertLess(float(forecast_loss(model(features),items,[4,8]).detach()),initial*.2)
        state=dict(model=model.state_dict(),feature_names=FEATURES,mean=torch.zeros(len(FEATURES)),
            std=torch.ones(len(FEATURES)),adopted='gate',constant_alpha=.5)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gate.pt'; torch.save(state,path); loaded=torch.load(path,weights_only=True)
            raw,deployed=gate_predict(loaded,features)
            np.testing.assert_allclose(raw,model(features).detach().numpy()); np.testing.assert_equal(raw,deployed)
        state['adopted']='constant'; raw,deployed=gate_predict(state,features)
        self.assertGreater(float(raw.std()),.1); np.testing.assert_equal(deployed,.5)

    def test_policy_requires_validation_and_zero_penalty_is_eligible(self):
        table=dict(split='validation',plain100=1.,diversity={str(s):1.+s for s in STRENGTHS},
            mixture={str(s):{str(a):(a-.5)**2+1 for a in ALPHAS} for s in STRENGTHS})
        policy=select_policy({'fixture':table})
        self.assertEqual(policy['diversity_weight'],0.); self.assertEqual(policy['analogue_alpha']['fixture'],.5)
        table['split']='test'
        with self.assertRaises(ValueError): select_policy({'fixture':table})


if __name__=='__main__': unittest.main()
