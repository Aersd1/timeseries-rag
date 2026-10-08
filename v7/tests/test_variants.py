import copy
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from uni2ts.model.moirai2 import Moirai2Module, Moirai2Forecast
from v6.common import write_json, save_torch
from v6.data import ingest, Windows
from v6.model import encoder_loss
from v7.common import config
from v7.model import MoiraiEncoder, load
from v7.moirai import MoiraiAdapter
from v7.retrieval import build as build_a, MoiraiRetriever
from v7.rerank import future_scores, V4MoiraiReranker, build as build_b
from v7.train import fit, FrozenFeatures, loss_for_batch
from v7.evaluate import evaluate


def tiny_adapter(dropout=.0):
    arch = dict(d_model=64, d_ff=128, num_layers=1, patch_size=16, max_seq_len=64,
        attn_dropout_p=dropout, dropout_p=dropout, scaling=True, num_predict_token=2,
        quantile_levels=[.1,.5,.9])
    return MoiraiAdapter(Moirai2Module(**arch), arch, dict(fixture=True, feature='last_context_token'))


def fixture(root):
    c = config(Path(__file__).parents[1]/'configs/server.json')
    c['length'], c['horizons'] = 40, [4,8,16]
    c['model'].update(hidden=8, dim=6, history_bins=6)
    c['training'].update(batch_size=8, samples_per_series=16, sample_stride=16, encoder_epochs=1)
    c['index'].update(leaf_size=16, fanout=4, shard_windows=256, batch_size=32, oversample=64, stride=8)
    c['evaluation'].update(queries_per_series=2, k=2, leaf_budgets=[0], audit_queries=0)
    c['rerank'].update(chunk_points=2000, max_points_per_series=10000, capacity=4000)
    x = np.sin(np.arange(10000)*.071)+.4*np.sin(np.arange(10000)*.023)
    csv=root/'data.csv'; np.savetxt(csv, x, delimiter=',', header='value', comments='')
    c['data']['sources']=[dict(glob=str(csv), columns=['value'], group='fixture')]
    path=root/'config.json'; write_json(path,c)
    meta=ingest(path,root/'store')
    return c,path,meta


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_latent_matches_official_context_and_freezes_dropout(self):
        adapter=tiny_adapter(.2)
        x=torch.randn(2,40)
        adapter.train()
        self.assertFalse(adapter.module.training)
        actual=adapter.encode(x)
        self.assertTrue(torch.equal(actual,adapter.encode(x)))
        forecast=Moirai2Forecast(module=adapter.module, prediction_length=16, context_length=40,
            target_dim=1,feat_dynamic_real_dim=0,past_feat_dynamic_real_dim=0).eval()
        inputs=forecast._convert(16,x[...,None],torch.ones_like(x[...,None],dtype=torch.bool),torch.zeros_like(x,dtype=torch.bool))
        captured=[]
        handle=adapter.module.encoder.register_forward_hook(lambda m,a,o:captured.append(o))
        try:
            with torch.no_grad(): adapter.module(*inputs,training_mode=False)
        finally: handle.remove()
        torch.testing.assert_close(actual,captured[0][:,2],rtol=1e-5,atol=1e-5)
        self.assertFalse(actual.requires_grad)

    def test_pinned_cache_load_needs_no_network(self):
        from safetensors.torch import save_model
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); adapter=tiny_adapter()
            write_json(root/'config.json',adapter.architecture)
            save_model(adapter.module,str(root/'model.safetensors'))
            with patch('huggingface_hub.snapshot_download',return_value=str(root)) as download:
                loaded=MoiraiAdapter.pretrained({})
                self.assertEqual(download.call_count,1)
                self.assertTrue(download.call_args.kwargs['local_files_only'])
                x=torch.randn(1,40)
                torch.testing.assert_close(adapter.encode(x),loaded.encode(x),rtol=0,atol=0)
            with self.assertRaises(ValueError): MoiraiAdapter.pretrained(dict(revision='main'))

    def test_forecast_handles_recursive_and_constant_context(self):
        adapter=tiny_adapter()
        x=torch.stack([torch.zeros(40),torch.arange(40,dtype=torch.float32)])
        y=adapter.forecast(x,65)  # More than 2 predicted tokens: exercises recursion.
        self.assertEqual(y.shape,(2,3,65))
        self.assertTrue(torch.isfinite(y).all())
        with self.assertRaises(ValueError): adapter.encode(torch.full((1,40),float('nan')))

    def test_encoder_gradients_and_history_only_input(self):
        c=config(Path(__file__).parents[1]/'configs/server.json')
        c['length'],c['horizons']=40,[4,8,16]
        adapter=tiny_adapter(.2); model=MoiraiEncoder(c,adapter).train()
        batch=dict(x=torch.randn(8,40),y=torch.randn(8,16),floor=torch.ones(8)*.1,
            sid=torch.arange(8),start=torch.zeros(8,dtype=torch.long))
        loss,_=encoder_loss(model,batch); loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None for p in adapter.parameters()))
        self.assertGreater(float(model.readout[0].weight.grad.abs().sum()),0)
        self.assertFalse(adapter.module.training)
        model.eval()
        with torch.no_grad():
            a=model(batch['x'],batch['floor'])['learned']
            batch['y']*=10000
            b=model(batch['x'],batch['floor'])['learned']
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_frozen_cache_preserves_loss_gradients_and_sample_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,path,_=fixture(Path(tmp))
            ds=Windows(Path(tmp)/'store',c,'train')
            try:
                adapter=tiny_adapter(); model=MoiraiEncoder(c,adapter).eval()
                cached=FrozenFeatures(ds,adapter,'cpu',batch_size=3)
                # Shuffle the cached dataset and include a repeated position.
                indices=[4,1,7,1,3,5]
                batch=torch.utils.data.default_collate([cached[i] for i in indices])
                torch.testing.assert_close(batch['moirai_signature'],adapter.encode(batch['x']),rtol=1e-5,atol=1e-5)
                raw=dict(batch);raw.pop('moirai_signature')
                first,_=loss_for_batch(model,raw);first.backward()
                gradients={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
                model.zero_grad(set_to_none=True)
                with patch.object(adapter,'encode',side_effect=AssertionError('Cache must avoid inference')):
                    second,_=loss_for_batch(model,batch);second.backward()
                torch.testing.assert_close(first,second,rtol=1e-5,atol=1e-5)
                for n,p in model.named_parameters():
                    if n in gradients: torch.testing.assert_close(gradients[n],p.grad,rtol=1e-4,atol=1e-5)
                self.assertIsNone(model.prior._signature)
                with self.assertRaises(RuntimeError):
                    with model.prior.cached_signature(batch['moirai_signature']):
                        raise RuntimeError('A failed loss must clear the override')
                self.assertIsNone(model.prior._signature)
            finally: ds.store.close()
        self.assertNotIn('y',inspect.signature(MoiraiRetriever.retrieve).parameters)
        self.assertNotIn('y',inspect.signature(V4MoiraiReranker.retrieve).parameters)


class ScoreTests(unittest.TestCase):
    def test_future_ranking_changes_order_without_true_query_future(self):
        q=np.array([np.full(8,-1),np.zeros(8),np.ones(8)])
        future=np.array([np.full(8,8),np.zeros(8),np.full(8,2)])
        score=future_scores(q,[.1,.5,.9],future,0,1,[4,8])
        self.assertEqual(np.argsort(score).tolist(),[1,2,0])
        # Same transformation for predicted/candidate curves preserves ranking and scores.
        transformed=future_scores(q*3+7,[.1,.5,.9],future*3+7,7,3,[4,8])
        np.testing.assert_allclose(score,transformed)
        # Verify against an independently written scalar pinball expression.
        expected=np.mean([np.mean([max(a*(8-v),(a-1)*(8-v)) for a,v in zip([.1,.5,.9],[-1,0,1])]) for _ in range(8)])
        self.assertAlmostEqual(score[0],expected)

    def test_quantile_crossing_and_bad_horizon(self):
        q=np.array([[1.,1.],[-1.,-1.],[0.,0.]])
        a=future_scores(q,[.1,.5,.9],np.zeros((1,2)),0,1,[2])
        b=future_scores(np.sort(q,axis=0),[.1,.5,.9],np.zeros((1,2)),0,1,[2])
        np.testing.assert_equal(a,b)
        with self.assertRaises(ValueError): future_scores(q,[.1,.5,.9],np.zeros((1,2)),0,1,[3])


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_a_training_index_query_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); c,cp,meta=fixture(root)
            with patch('v7.train.MoiraiAdapter.pretrained',return_value=tiny_adapter()):
                fit(root/'store',cp,root/'train')
            checkpoint=root/'train/best.pt'
            m,state=load(checkpoint)
            self.assertEqual(state['data_id'],meta['data_id'])
            built=build_a(root/'store',checkpoint,root/'index')
            self.assertTrue(built['complete'])
            ds=Windows(root/'store',c,'test'); item=ds[0]; ds.store.close()
            r=MoiraiRetriever(root/'store',checkpoint,root/'index')
            try:
                result=r.retrieve(item['x'],int(item['sid']),int(item['start']))
                self.assertEqual(result['prediction'].shape,(16,))
                self.assertTrue(np.isfinite(result['prediction']).all())
                for hit in result['hits']:
                    self.assertLessEqual(hit['history_nmse'],c['evaluation']['history_max_nmse'])
                    self.assertLessEqual(hit['start']+56,r.store.series[0]['memory_end'])
                with self.assertRaises(ValueError): r.retrieve(item['x'],0,0)
            finally: r.close()
            report=evaluate(root/'store',cp,root/'eval',checkpoint_a=checkpoint,index_a=root/'index')
            self.assertTrue(report['forecast'])
            manifest=root/'index/manifest.json'
            payload=__import__('json').loads(manifest.read_text()); payload['checkpoint_sha256']='bad'; write_json(manifest,payload)
            with self.assertRaises(ValueError): MoiraiRetriever(root/'store',checkpoint,root/'index')

    def test_b_original_v4_one_call_then_rerank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); c,cp,_=fixture(root)
            build_b(root/'store',cp,root/'index')
            ds=Windows(root/'store',c,'test'); item=ds[0]; ds.store.close()
            engine=V4MoiraiReranker(root/'store',root/'index',adapter=tiny_adapter())
            native=engine.engine(0)
            name='search_baseline' if hasattr(native,'search_baseline') else 'search'
            try:
                with patch.object(native,name,wraps=getattr(native,name)) as search:
                    result=engine.retrieve(item['x'],0,int(item['start']))
                    self.assertEqual(search.call_count,1)
                    self.assertEqual(search.call_args.kwargs['k'],20)
                self.assertEqual(len(result['candidates']),20)
                self.assertEqual(len(result['hits']),5)
                self.assertTrue(np.isfinite(result['prediction']).all())
                self.assertEqual([h['rerank_score'] for h in result['candidates']], sorted(h['rerank_score'] for h in result['candidates']))
                for hit in result['candidates']:
                    self.assertLessEqual(hit['start']+56,engine.store.series[0]['memory_end'])
                with self.assertRaises(ValueError): engine.retrieve(item['x'],0,0)
                original=getattr(native,name)(item['x'],16,engine.store.series[0],int(item['start']),
                    'same_series',capacity=4000,k=20)
                partial=dict(original,episodes=original['episodes'][:3])
                with patch.object(native,name,return_value=partial) as search:
                    short=engine.retrieve(item['x'],0,int(item['start']))
                    self.assertEqual(search.call_count,1)
                    self.assertEqual(len(short['hits']),3)
                    self.assertFalse(short['stats']['complete_topk'])
                    self.assertAlmostEqual(sum(h['weight'] for h in short['hits']),1.)
                # Empty pool is explicit and falls back to persistence, without refill.
                empty=dict(ordinary=[],episodes=[],episode_complete=True,retrieval_ms=0.)
                with patch.object(native,name,return_value=empty) as search:
                    result=engine.retrieve(item['x'],0,int(item['start']))
                    self.assertEqual(search.call_count,1)
                    self.assertFalse(result['stats']['complete_topk'])
                    np.testing.assert_equal(result['prediction'],np.full(16,item['x'][-1]))
            finally: engine.close()


if __name__=='__main__': unittest.main()
