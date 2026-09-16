import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from v5.common import read_json, write_json, valid_ranges, sample_ranges, affine, norm_scale, validate_config
from v5.data import finite_runs, ingest, Store
from v5.teacher import generate, Labels


def config():
    c = read_json(Path(__file__).resolve().parents[1]/'configs'/'server.json')
    c['length'] = 64; c['horizons'] = [4,8,12]
    c['teacher'].update(neighbors=16,shard_queries=2,queries_per_series=dict(train=3,validation=2,test=2))
    c['training'].update(candidates=8,batch_size=2,epochs=1,amp=False)
    c['model'].update(dim=8,hidden=16,vocab=16)
    c['index'].update(stride=16,batch_size=16)
    c['evaluation'].update(topk=[3,10],candidate_fraction=.1,analog_k=3)
    c['data']['chunk_rows'] = 137
    return c


def fixture(root):
    c = config(); x = np.sin(np.arange(3000)*.027)+.05*np.cos(np.arange(3000)*.19)
    frame = pd.DataFrame(dict(timestamp=np.arange(len(x)),value=x))
    csv = Path(root)/'input.csv'; frame.to_csv(csv,index=False)
    c['data']['sources'] = [dict(glob=str(csv),columns=['value'],time_column='timestamp',time_kind='numeric',expected_interval=1,group='synthetic')]
    cfg = Path(root)/'config.json'; write_json(cfg,c)
    store_path = Path(root)/'store'; ingest(cfg,store_path)
    return c,store_path


class NumpyOracle:
    """Test-only brute force. No model fitting, optimizer or production training."""
    def __init__(self,store,sid,c):
        self.store,self.sid,self.c = store,sid,c
        self.idx = type('I',(),{'manifest':{'series':[dict(start=0)]}})()
    def search(self,q,h,meta,start,scope,capacity,k):
        m = len(q); last = meta['memory_end']-m-h+1
        x = np.asarray(self.store.values(self.sid)[:meta['memory_end']],dtype=float)
        windows = np.lib.stride_tricks.sliding_window_view(x,m)[:last]
        def z(x):
            xc = x-x.mean(axis=-1,keepdims=True); return xc/np.maximum(x.std(axis=-1,keepdims=True),1e-10)
        ds = np.sum((z(windows)-z(q))**2,axis=-1)
        order = np.argsort(ds,kind='stable')[:k]
        return dict(ordinary=[(float(ds[p]),0,int(p)) for p in order],retrieval_ms=0.)
    def close(self): pass


def labels_fixture(root,store_path,c):
    output = Path(root)/'teacher'
    with patch('v5.teacher.make_oracle',side_effect=lambda store,sid,c,dest:NumpyOracle(store,sid,c)):
        generate(store_path,output)
    return output


class CoreTests(unittest.TestCase):
    def test_finite_runs_chunk_edges(self):
        x = np.array([1,2,np.nan,4,5,6,np.nan,np.inf,8,9.])
        self.assertEqual(finite_runs(x,3),[[0,2],[3,6],[8,10]])
        self.assertEqual(finite_runs(np.ones(20),3),[[0,20]])

    def test_split_ranges_and_affine(self):
        s = dict(runs=[[0,200],[210,500]])
        positions = sample_ranges(valid_ranges(s,64,12,100,400,76),100,np.random.default_rng(1))
        self.assertTrue(all(p >= 100 and p+76 <= 400 for p in positions))
        self.assertFalse(any(p < 210 and p+76 > 200 for p in positions))
        x = np.arange(64.); y = np.arange(64,76.)[None]
        np.testing.assert_allclose(affine(3*x+2,x[None],y),3*y+2)
        self.assertGreater(norm_scale(np.ones(10),0.),0)
        c = config(); c['split']['memory'] = .8
        with self.assertRaises(ValueError): validate_config(c)

    def test_ingestion_hash_split_and_label_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            c,path = fixture(temp); store = Store(path)
            self.assertEqual(store.series[0]['memory_end'],750)
            self.assertEqual(store.series[0]['runs'],[[0,3000]])
            self.assertEqual(len(store.series[0]['values_sha256']),64)
            teacher = labels_fixture(temp,path,c)
            for split,low,high in [('train',750,1800),('validation',1800,2400),('test',2400,3000)]:
                labels = Labels(teacher,split); self.assertGreater(len(labels),0)
                for i in range(len(labels)):
                    row = labels[i]; self.assertGreaterEqual(row['start'],low)
                    self.assertLessEqual(row['start']+76,high)
                    self.assertTrue(np.all(row['candidates'][row['valid']]+76 <= 750))
                    self.assertNotIn(row['start'],row['candidates'])
                    self.assertTrue(np.isfinite(row['future_nmse']).all())
            # Resume completed labels without recomputing the oracle.
            with patch('v5.teacher.make_oracle',side_effect=AssertionError('unexpected recompute')):
                generate(path,teacher)

    def test_timestamp_reversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            c,path = fixture(temp)
            frame = pd.read_csv(Path(temp)/'input.csv'); frame.loc[200,'timestamp']=0
            frame.to_csv(Path(temp)/'input.csv',index=False)
            with self.assertRaises(ValueError): ingest(Path(temp)/'config.json',Path(temp)/'bad')


if __name__ == '__main__': unittest.main()
