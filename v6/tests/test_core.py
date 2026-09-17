import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.stats import norm
from v6.common import config, write_json, save_torch
from v6.model import FuturePrior, BeliefEncoder, prior_loss, posterior, encoder_loss
from v6.index import Library, boxes, diverse, lower_bound, build
from v6.data import ingest, Windows
from v6.inference import Retriever
from v6.evaluate import evaluate, probabilistic_scores
from v6.analyze import analyze
from v6.compare import compare


def small_config():
    c = config(Path(__file__).parents[1]/'configs/server.json')
    c['length'], c['horizons'] = 40, [4, 8, 16]
    c['model'].update(hidden=8, dim=6, components=2, belief_bins=3, history_bins=6)
    c['training'].update(batch_size=8, samples_per_series=12, sample_stride=8)
    c['index'].update(leaf_size=16, fanout=4, shard_windows=128, batch_size=32, oversample=8)
    c['evaluation'].update(queries_per_series=2, k=2, leaf_budgets=[0, 2], audit_queries=1, bootstrap_samples=10, max_examples=1)
    return c


class ProbabilityTests(unittest.TestCase):
    def test_bayes_and_likelihood(self):
        p = dict(center=torch.tensor([0.]), scale=torch.tensor([1.]), logp=torch.log(torch.tensor([[.3,.7]])),
                 mu=torch.tensor([[[0.,0.],[2.,2.]]]), sigma=torch.ones(1,2,2))
        target = torch.tensor([[1.5,1.5]])
        likelihood = np.array([norm.pdf(1.5, 0)**2*.3, norm.pdf(1.5, 2)**2*.7])
        np.testing.assert_allclose(posterior(p,target).numpy()[0], likelihood/likelihood.sum(), rtol=1e-6)
        self.assertAlmostEqual(float(prior_loss(p,target)), -np.log(likelihood.sum())/2, places=6)
        single=.3*norm.pdf(1.5,0)+.7*norm.pdf(1.5,2)
        self.assertAlmostEqual(float(prior_loss(p,target,[1,2])),(-np.log(single)-np.log(likelihood.sum())/2)/2,places=6)

    def test_normal_calibration(self):
        p = dict(center=0., scale=1., logp=np.array([0.]), mu=np.zeros((1,8)), sigma=np.ones((1,8)))
        records, interval = probabilistic_scores(p, np.zeros(8), [4,8])
        self.assertAlmostEqual(records[0]['crps'], (np.sqrt(2)-1)/np.sqrt(np.pi), places=7)
        self.assertAlmostEqual(interval[0][0], norm.ppf(.05), places=6)
        self.assertEqual(records[0]['coverage90'], 1)

    def gradients(self, device, amp):
        c = small_config(); torch.manual_seed(17)
        x = torch.randn(8,40,device=device); x[0] = 0
        batch = dict(x=x, y=torch.randn(8,16,device=device), floor=torch.full((8,),.1,device=device),
                     sid=torch.arange(8,device=device), start=torch.zeros(8,dtype=torch.long,device=device))
        prior = FuturePrior(c).to(device)
        with torch.autocast(device_type=device, enabled=amp):
            loss = prior_loss(prior(x,batch['floor']),batch['y'])
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in prior.parameters() if p.grad is not None))
        model = BeliefEncoder(c).to(device)
        before = {k: v.clone() for k,v in model.state_dict().items()}
        with torch.autocast(device_type=device, enabled=amp):
            loss, parts = encoder_loss(model,batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None for p in model.prior.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(float(parts['usable_fraction']), 0)
        for key,value in model.state_dict().items():
            self.assertTrue(torch.equal(value,before[key]))  # No optimizer step/local training.
        model.eval()
        with torch.no_grad():
            a = model(batch['x'],batch['floor'])['learned']
            batch['y'] *= 1000
            b = model(batch['x'],batch['floor'])['learned']
        self.assertTrue(torch.equal(a,b))

    def test_cpu_gradients_and_no_future_input(self):
        self.gradients('cpu',False)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
    def test_cuda_amp_gradients(self):
        self.gradients('cuda',True)

    def test_overlapping_pair_mask(self):
        c = small_config(); model = BeliefEncoder(c)
        batch = dict(x=torch.randn(4,40), y=torch.randn(4,16), floor=torch.full((4,),.1),
                     sid=torch.zeros(4,dtype=torch.long), start=torch.arange(4))
        loss, parts = encoder_loss(model,batch)
        self.assertEqual(float(parts['usable_fraction']),0.)
        self.assertEqual(float(parts['pair']),0.)
        self.assertTrue(torch.isfinite(loss))


class HierarchyTests(unittest.TestCase):
    def test_matches_exhaustive_and_exclusions(self):
        rng = np.random.default_rng(44)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); c = small_config(); c['index']['channels']=['learned']
            shards=[]; all_values=[]
            for sid in range(2):
                path=root/f'{sid:06d}'; (path/'learned').mkdir(parents=True)
                values=rng.normal(size=(129,6)).astype('f4'); values[1]=values[0]
                starts=np.arange(129,dtype=np.int64)*40
                np.save(path/'starts.npy',starts); np.save(path/'learned/vectors.npy',values)
                levels=boxes(path/'learned',values,16,4)
                shards.append(dict(path=path.name,sid=sid,group='a',n=len(values),levels={'learned':levels}))
                all_values.extend((sid,int(p),v) for p,v in zip(starts,values))
            write_json(root/'manifest.json',dict(version=6,complete=True,config=c,shards=shards))
            library=Library(root)
            for query in [rng.normal(size=6).astype('f4') for _ in range(10)]+[all_values[0][2]]:
                actual,stats=library.search(query,k=5,query_sid=0,query_start=0,capacity=20)
                expected=sorted([dict(sid=s,start=p,distance=float(np.sum((v.astype(float)-query.astype(float))**2)))
                                 for s,p,v in all_values if s!=0 or p>=56],key=lambda r:(r['distance'],r['sid'],r['start']))
                expected=diverse(expected[:20],5,40)
                self.assertEqual(actual,expected)
                self.assertTrue(stats['exact_diverse_topk'])
            _,stats=library.search(np.zeros(6),k=5,leaf_budget=1,capacity=20)
            self.assertFalse(stats['certified'])
            self.assertLessEqual(stats['leaves'],1)
            _,stats=library.search(np.zeros(6),k=5,time_budget_ms=1e-9)
            self.assertFalse(stats['certified'])
            library.close()

    def test_bounds(self):
        rng=np.random.default_rng(2)
        for magnitude in (1e-8,1,1e5):
            v=(rng.normal(size=(64,7))*magnitude).astype('f4'); q=rng.normal(size=7)*magnitude
            bound=lower_bound(q,np.nextafter(v.min(0),np.float32(-np.inf)),np.nextafter(v.max(0),np.float32(np.inf)))
            self.assertLessEqual(bound,np.sum((v.astype(float)-q)**2,axis=1).min())

    def test_underfilled_is_reported(self):
        hits=[dict(sid=0,start=i,distance=float(i)) for i in range(8)]
        self.assertEqual(len(diverse(hits,4,40)),1)


class PipelineTests(unittest.TestCase):
    def test_smoke_no_training(self):
        # Synthetic fixtures, random checkpoint, NO fit() or optimizer updates.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); c=small_config()
            x=np.sin(np.arange(2000)*.07)+np.arange(2000)*.001
            x[123]=np.nan
            csv=root/'input.csv'; np.savetxt(csv,x,delimiter=',',header='value',comments='')
            c['data']['sources']=[dict(glob=str(csv),columns=['value'],group='test')]
            cp=root/'config.json'; write_json(cp,c); meta=ingest(cp,root/'store')
            for split in ('prior','train','validation','test'):
                ds=Windows(root/'store',c,split)
                s=ds.store.series[0]
                bounds={'prior':(0,s['memory_end']),'train':(s['memory_end'],s['train_end']),
                        'validation':(s['train_end'],s['validation_end']),'test':(s['validation_end'],s['n'])}[split]
                for i in range(len(ds)):
                    item=ds[i]; self.assertGreaterEqual(item['start'],bounds[0]); self.assertLessEqual(item['start']+56,bounds[1])
                    self.assertTrue(np.isfinite(item['x']).all())
                ds.store.close()
            torch.manual_seed(19); model=BeliefEncoder(c)
            checkpoint=root/'fixture.pt'
            save_torch(checkpoint,dict(version=6,stage='encoder',config=c,model=model.state_dict(),data_id=meta['data_id'],epoch=-1))
            index=build(root/'store',checkpoint,root/'index')
            self.assertTrue(index['complete'])
            evaluate(root/'store',checkpoint,root/'index',root/'evaluation')
            report=analyze(root/'evaluation')
            self.assertGreater(len(report['forecast']),0)
            self.assertTrue((root/'evaluation/analysis_bundle.zip').exists())
            import zipfile
            with zipfile.ZipFile(root/'evaluation/analysis_bundle.zip') as bundle:
                self.assertFalse(any('private' in name for name in bundle.namelist()))
                self.assertNotIn(str(csv),bundle.read('run_public.json').decode())
            comparison=compare(root/'evaluation',root/'evaluation',root/'comparison.json')
            self.assertTrue(all(row['paired_improvement']==0 for row in comparison))
            self.assertIn('embedding_recall_against_full',report)
            self.assertTrue(all(v['mean']==1 for k,v in report['embedding_recall_against_full'].items() if k.endswith('leaves0')))
            metrics=[json.loads(line) for line in (root/'evaluation/metrics.jsonl').read_text().splitlines()]
            self.assertTrue(all(np.isfinite(row['nmse']) for row in metrics))
            from v6.fusion import calibrate, apply
            evaluate(root/'store',checkpoint,root/'index',root/'validation',split='validation')
            fitted=calibrate(root/'validation',root/'fusion.json')
            r=Retriever(root/'store',checkpoint,root/'index',fusion_path=root/'fusion.json')
            ds=Windows(root/'store',c,'test'); b=ds[0]
            found=r.retrieve(b['x'],int(b['sid']),int(b['start']))
            self.assertTrue(np.isfinite(found['prediction']).all())
            self.assertEqual(len(found['raw_prediction']),16)
            ds.store.close(); r.close()
            # Joint retrieval certifies its raw-history threshold, not real future similarity.
            r=Retriever(root/'store',checkpoint,root/'index')
            found=r.retrieve(b['x'],int(b['sid']),int(b['start']),channel='joint')
            self.assertGreater(len(found['hits']),0)
            for hit in found['hits']:
                self.assertLessEqual(hit['history_nmse'],c['evaluation']['history_max_nmse'])
                self.assertGreaterEqual(abs(hit['start']-int(b['start'])),56)
            r.close()
            # A different weights file must not query an existing index.
            save_torch(root/'wrong.pt',dict(version=6,stage='encoder',config=c,model=BeliefEncoder(c).state_dict(),data_id=meta['data_id'],epoch=-1))
            with self.assertRaisesRegex(ValueError,'different weights'):
                Retriever(root/'store',root/'wrong.pt',root/'index')


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
