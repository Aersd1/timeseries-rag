import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
from test_core import config,fixture,labels_fixture

HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH,'PyTorch unavailable: run these CPU forward/backward tests on server/CI')
class ModelTests(unittest.TestCase):
    def test_token_metadata_causality_and_padding(self):
        import torch
        from v5.model import UnifiedTokens
        c = config(); c['model']['primitive']=8; model = UnifiedTokens(c).eval()
        x = torch.randn(2,244); y = x.clone(); y[:,16:] += 99
        a,b = model(x),model(y)
        self.assertEqual(tuple(a['ids'].shape),(2,31))
        self.assertTrue(torch.equal(a['duration'][:,-1],torch.tensor([4,4])))
        self.assertTrue(torch.equal(a['ids'][:,:2],b['ids'][:,:2]))
        self.assertTrue(torch.isfinite(model(torch.ones(2,244))['prediction']).all())
        self.assertTrue(torch.all(a['ids'] >= 0) and torch.all(a['ids'] < c['model']['vocab']))

    def test_losses_backward_no_optimizer(self):
        import torch
        from v5.model import UnifiedTokens,objective,distill
        from v5.train import TokenDataset
        from torch.utils.data import DataLoader
        torch.manual_seed(123)
        with tempfile.TemporaryDirectory() as temp:
            c,path=fixture(temp); teacher=labels_fixture(temp,path,c)
            dataset=TokenDataset(path,teacher,'train',c)
            batch=next(iter(DataLoader(dataset,batch_size=2)))
            model=UnifiedTokens(c); loss,parts=objective(model,batch)
            self.assertEqual(set(parts),{'rate','reconstruction','retrieval','future','forecast','vq'})
            loss.backward()  # Gradient validation only. NO optimizer/training step.
            for prefix in ('encoder','codebook','shape_head','predictive_head','forecaster'):
                grads=[p.grad for n,p in model.named_parameters() if n.startswith(prefix)]
                self.assertTrue(any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads),prefix)
            teacher_p=torch.tensor([[.3,.7,0.]])
            kl=distill(teacher_p,torch.tensor([[np.log(.3),np.log(.7),0.]],dtype=torch.float32),torch.tensor([[True,True,False]]))
            self.assertAlmostEqual(float(kl),0.,places=6)
            dataset.store.close()

    def test_inference_index_evaluation_bundle_without_training(self):
        import torch
        import zipfile
        from v5.model import UnifiedTokens
        from v5.token_index import build,TokenLibrary
        from v5.evaluate import evaluate
        from v5.analyze import analyze
        from v5.common import read_json
        from v5.data import Store
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); c,path=fixture(root); teacher=labels_fixture(root,path,c)
            model=UnifiedTokens(c).eval(); checkpoint=root/'untrained.pt'
            torch.save(dict(model=model.state_dict(),config=c,data_id=Store(path).meta['data_id'],epoch=-1),checkpoint)
            index=root/'index'; build(path,checkpoint,index)
            lib=TokenLibrary(index); lib.select(0)
            vector=np.array(lib.shape[0]); expected=np.argsort(-(lib.shape @ vector),kind='stable')[:5]
            np.testing.assert_array_equal(lib.cosine(vector,5),expected)
            lib.close()
            output=root/'evaluation'; evaluate(path,teacher,checkpoint,index,output)
            result=analyze(output)
            self.assertFalse(result['actual_training_results'])
            self.assertTrue(all(0 <= r['recall'] <= 1 for r in result['retrieval']))
            with zipfile.ZipFile(output/'analysis_bundle.zip') as archive:
                self.assertIn('analysis.json',archive.namelist())
                self.assertNotIn('examples_private.json',archive.namelist())
                self.assertFalse(any(n.endswith('.pt') or n.endswith('.bin') for n in archive.namelist()))


if __name__ == '__main__': unittest.main()
