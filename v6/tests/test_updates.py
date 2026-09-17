import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from v6.model import signature, CDF_GRID
from v6.index import Candidates, Library, boxes, repack
from v6.fusion import FusionStats, calibrate, apply, check
from v6.common import write_json, read_json
from v6.sampling import HistoryBatches
from test_core import small_config


class UpdateTests(unittest.TestCase):
    def test_history_batches_never_read_future_and_are_reproducible(self):
        from types import SimpleNamespace
        c=small_config(); refs=[(0,j*100) for j in range(40)]; requested=[]
        class Store:
            series=[dict(memory_std=1.)]
            def window(self,sid,start,length):
                requested.append(length)
                if length!=c['length']: raise AssertionError('Sampler read future')
                return np.sin(np.arange(length)*.15+(start//100)%5).astype('f4')
        class Dataset:
            def __init__(self): self.c,self.refs,self.store=c,refs,Store()
            def __len__(self): return len(refs)
        sampler=HistoryBatches(Dataset(),8,12)
        first=list(sampler); self.assertEqual(first,list(sampler))
        self.assertTrue(all(len(batch)==8 and len(set(batch))==8 for batch in first))
        self.assertTrue(all(length==40 for length in requested))
        sampler.epoch=1; self.assertNotEqual(first,list(sampler))

    def test_joint_repack_aligns_independent_spatial_orders(self):
        rng=np.random.default_rng(42); c=small_config(); c['index']['channels']=['learned','history']
        n=97; base=rng.normal(size=(n,6)).astype('f4'); hist=rng.normal(size=(n,6)).astype('f4')
        starts=np.arange(n,dtype=np.int64)*40
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); old=root/'old'; shard=old/'000000'; shard.mkdir(parents=True)
            levels={}
            for channel,v in [('learned',base),('history',hist)]:
                order=rng.permutation(n); cp=shard/channel; cp.mkdir()
                np.save(cp/'vectors.npy',v[order]); np.save(cp/'starts.npy',starts[order])
                levels[channel]=boxes(cp,v[order],16,4)
            np.save(shard/'starts.npy',starts)
            write_json(old/'manifest.json',dict(version=6,complete=True,layout='spatial',windows=n,config=c,
                 shards=[dict(path='000000',sid=0,group='test',n=n,levels=levels)]))
            repack(old,root/'joint',['joint'],joint_history_weight=.3,history_max_nmse=.5)
            lib=Library(root/'joint')
            all_vectors=np.concatenate([np.float32(np.sqrt(.7))*base,np.float32(np.sqrt(.3))*hist],axis=1)
            q=all_vectors[20]
            actual,_=lib.search(q,channel='joint',k=4,capacity=20)
            ds=((all_vectors.astype(float)-q.astype(float))**2).sum(1)
            order=np.lexsort((starts,ds))[:4]
            self.assertEqual([hit['start'] for hit in actual],starts[order].tolist())
            lib.close()

    def test_signature_invariant_to_regime_labels(self):
        torch.manual_seed(33)
        p=torch.randn(5,4).log_softmax(-1); mu=torch.randn(5,4,16); sd=torch.rand(5,4,16)+.1
        permutation=[2,0,3,1]
        a=signature(p,mu,sd,3,'cdf')
        b=signature(p[:,permutation],mu[:,permutation],sd[:,permutation],3,'cdf')
        self.assertEqual(a.shape,(5,3*(len(CDF_GRID)+2)))
        torch.testing.assert_close(a,b)
        self.assertFalse(torch.allclose(signature(p,mu,sd,3),signature(p[:,permutation],mu[:,permutation],sd[:,permutation],3)))

    def test_patchtst_direct_forecast_uses_history_only(self):
        from v6.patchtst_direct import PatchTSTForecast, forecast_loss
        c=small_config()
        c['model'].update(backbone='patchtst', patch_len=8, patch_stride=4, n_heads=2, e_layers=1, dropout=0.)
        model=PatchTSTForecast(c)
        x=torch.randn(6,40); floor=torch.full((6,),.1); y=torch.randn(6,16)
        out=model(x,floor)
        self.assertEqual(tuple(out['raw'].shape),(6,16))
        loss,_=forecast_loss(model,dict(x=x,y=y,floor=floor))
        self.assertTrue(torch.isfinite(loss))
        model.eval()
        with torch.no_grad():
            a=model(x,floor)['raw']
            b=model(x,floor)['raw']
        torch.testing.assert_close(a,b)

    def test_score_prediction_matches_evaluate_denominators(self):
        from v6.patchtst_direct import instance_scale, score_prediction
        x=np.array([0.,2.,4.,6.],dtype=float); y=np.array([8.,10.],dtype=float)
        pred=np.array([7.,11.],dtype=float); floor=0.5; memory_std=2.
        scale=instance_scale(x,floor)
        self.assertAlmostEqual(scale, float(x.std()))
        self.assertEqual(instance_scale(np.zeros(4),floor), floor)
        row=score_prediction(pred,y,x,floor,memory_std,[2],0,0,'m')[0]
        z_pred=(pred-x.mean())/scale; z_y=(y-x.mean())/scale
        mse_z=float(np.mean((z_pred-z_y)**2))
        self.assertAlmostEqual(row['mse_z'], mse_z)
        self.assertAlmostEqual(row['history_nmse'], mse_z)
        self.assertAlmostEqual(row['mse'], float(np.mean((pred-y)**2)))
        self.assertAlmostEqual(row['nmse'], row['mse']/memory_std**2)

    def test_bounded_candidates_global_ties(self):
        rng=np.random.default_rng(36); best=Candidates(31)
        ds=rng.integers(0,8,600).astype(float); ss=rng.integers(0,5,600); starts=np.arange(600)
        for sid in range(5):
            ids=np.flatnonzero(ss==sid)
            for a in range(0,len(ids),19):
                ii=ids[a:a+19]; best.add(ds[ii],sid,starts[ii]); self.assertLessEqual(len(best),31)
        expected=np.lexsort((starts,ss,ds))[:31]
        self.assertEqual(best.hits(),[dict(distance=float(ds[i]),sid=int(ss[i]),start=int(starts[i])) for i in expected])

    def test_repack_preserves_search_positions_and_repack_again(self):
        rng=np.random.default_rng(15); c=small_config(); c['index']['channels']=['learned']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'old'; cp=source/'000000/learned'; cp.mkdir(parents=True)
            values=rng.normal(size=(513,6)).astype('f4'); values[:3]=0
            starts=np.arange(513,dtype=np.int64)*40
            np.save(cp/'vectors.npy',values); np.save(cp.parent/'starts.npy',starts)
            levels=boxes(cp,values,16,4)
            write_json(source/'manifest.json',dict(version=6,complete=True,config=c,windows=len(starts),
                shards=[dict(path='000000',sid=0,group='test',n=len(starts),levels={'learned':levels})]))
            repack(source,root/'new'); repack(root/'new',root/'again')
            libraries=[Library(p) for p in (source,root/'new',root/'again')]
            try:
                for q in list(rng.normal(size=(6,6)).astype('f4'))+[np.zeros(6)]:
                    results=[lib.search(q,k=4,capacity=40,query_sid=0,query_start=80) for lib in libraries]
                    self.assertEqual(results[0][0],results[1][0]); self.assertEqual(results[0][0],results[2][0])
                    self.assertTrue(all(r[1]['exact_diverse_topk'] for r in results))
            finally:
                for lib in libraries: lib.close()

    def test_fusion_validation_only_and_policy_bound(self):
        c=small_config(); target=np.linspace(0,1,16)
        forecasts=dict(persistence=np.zeros(16),encoder_forecast=np.ones(16)*3,
                       probabilistic_prior=np.ones(16)*2,learned_leaves0=target)
        stats=FusionStats(c['horizons']); stats.add(forecasts,target,1.)
        run=dict(split='test',complete=True,data_id='d',checkpoint_sha256='w',index_sha256='i',config=c)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); write_json(root/'fusion_stats.json',stats.export(run))
            with self.assertRaisesRegex(ValueError,'VALIDATION'): calibrate(root,root/'fusion.json')
            run['split']='validation'; write_json(root/'fusion_stats.json',stats.export(run))
            fitted=calibrate(root,root/'fusion.json')
            check(fitted,'d','w','i',c,'learned_leaves0')
            np.testing.assert_allclose(apply(fitted,forecasts),target,atol=1e-6)
            with self.assertRaisesRegex(ValueError,'index_sha256'): check(fitted,'d','w','other',c,'learned_leaves0')
            with self.assertRaisesRegex(ValueError,'method'): check(fitted,'d','w','i',c,'learned_leaves128')

    def test_patchtst_backbone_shapes_and_no_future_input(self):
        from v6.model import make_backbone, BeliefEncoder, encoder_loss
        c=small_config()
        c['model'].update(backbone='patchtst', patch_len=8, patch_stride=4, n_heads=2, e_layers=1, dropout=0.)
        x=torch.randn(6,40); net=make_backbone(c)
        self.assertEqual(tuple(net(x).shape),(6,c['model']['hidden']))
        model=BeliefEncoder(c)
        batch=dict(x=x,y=torch.randn(6,16),floor=torch.full((6,),.1),
                   sid=torch.arange(6),start=torch.arange(6)*80)
        loss,_=encoder_loss(model,batch)
        self.assertTrue(torch.isfinite(loss))
        model.eval()
        with torch.no_grad():
            a=model(batch['x'],batch['floor'])['learned']
            batch['y']=batch['y']*100
            b=model(batch['x'],batch['floor'])['learned']
        self.assertTrue(torch.equal(a,b))


if __name__=='__main__': unittest.main()
