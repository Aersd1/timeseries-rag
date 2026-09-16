import tempfile
import unittest
import json
import zipfile
from pathlib import Path
from v5.common import write_json
from v5.analyze import analyze
from test_core import config


class AnalysisTests(unittest.TestCase):
    def test_synthetic_metrics_bundle_without_neural_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp); c=config()
            write_json(out/'run.json',dict(complete=True,queries=4,split='test',training_completed=False,config=c,
                compression=dict(points_per_token=4.,raw_to_token_payload_ratio=16/11,index_total_bytes=10000,
                                 used_codes=8,codebook_size=16,codebook_perplexity=7.5)))
            rows=[]; retrieval=[]; timing=[]
            for i in range(4):
                for method,error in [('persistence',2.),('raw_shape',1.),('token_forecaster',1.5)]:
                    rows.append(dict(query_id=str(i),sid=i%2,group='synthetic',horizon=4,method=method,
                        mse=error,mae=error,nmse=error,returned=3,mean_candidate_future_nmse=None))
                for route in ('cosine','bm25'):
                    retrieval.append(dict(query_id=str(i),sid=i%2,route=route,k=10,candidate_fraction=.1,
                        oracle_recall=.8,oracle_coverage=.9,teacher_tie_at10=False))
                    timing.append(dict(query_id=str(i),route=route,ranker='exact',total_ms=12.))
            for name,values in [('forecasts',rows),('retrieval',retrieval)]:
                (out/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in values),encoding='utf-8')
            write_json(out/'timing.json',timing)
            result=analyze(out)
            self.assertFalse(result['actual_training_results'])
            row=next(r for r in result['forecasts'] if r['group']=='ALL' and r['method']=='raw_shape')
            self.assertAlmostEqual(row['skill'],.5)
            with zipfile.ZipFile(out/'analysis_bundle.zip') as z:
                self.assertNotIn('examples_private.json',z.namelist())
                public=json.loads(z.read('run_public.json'))
                self.assertNotIn('sources',public['config']['data'])


if __name__ == '__main__': unittest.main()
