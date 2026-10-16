"""Rebuild sparse-cohort inputs and scores without refitting or reforecasting."""
import argparse
from pathlib import Path
import numpy as np
from v5.data import Store
from .common import sha256,write_json
from .memory_access import CONTEXT,align_pack,select_clock,history_controls
from .benchmark_memory_access import read,score


def verify(root):
    root=Path(root);frozen=read(root/'frozen.json');records=[]
    for name,h in frozen.items():
        folder=root/name
        if sha256(folder/'cohort.json')!=h:raise ValueError('Changed cohort definition')
        meta=read(folder/'cohort.json')
        if sha256(folder/'inputs.npz')!=meta['inputs_sha256']:raise ValueError('Changed model inputs')
        with np.load(folder/'inputs.npz',allow_pickle=False) as z:
            d={k:z[k].copy() for k in z.files}
        with np.load(folder/'predictions.npz',allow_pickle=False) as z:
            pred={k:z[k].copy() for k in z.files}
        result=read(folder/'result.json')
        for key,expected in result['metrics'].items():
            actual=score(d['raw_y'],pred[key],d['train_std'])
            for metric in actual:np.testing.assert_allclose(actual[metric],expected[metric],rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(pred['sid'],d['sid']);np.testing.assert_array_equal(pred['start'],d['start'])
        ids=[0,len(d['sid'])-1]
        if name=='electricity':ids.append(int(np.flatnonzero(d['sid']==313)[0]))
        store=Store(meta['store'])
        try:
            for row in ids:
                sid=int(d['sid'][row]);t=int(d['start'][row]);values=store.values(sid)
                long=np.array(values[t+96-CONTEXT:t+96],dtype=np.float32,copy=True)
                np.testing.assert_array_equal(long,d['moirai_2016'][row])
                np.testing.assert_array_equal(long[-512:],d['moirai_512'][row])
                np.testing.assert_array_equal(values[t+96:t+192],d['raw_y'][row])
                limit=int(meta['limits'][str(sid)])
                starts=np.array([p for a,b in store.series[sid]['runs'] for p in range(a,min(b,limit)-192+1)])
                for arm in ('real','random','periodic','recent','univariate'):
                    selected=d['selected_'+arm][row]
                    assert len(np.unique(selected))==10 and np.all(selected+192<=min(t,limit))
                    if arm in ('recent','periodic'):
                        np.testing.assert_array_equal(selected,select_clock(starts,t,limit,meta['period'],arm))
                    if arm=='univariate':
                        np.testing.assert_array_equal(selected,history_controls(values,starts,long[-96:],t,limit))
                    past=np.asarray(values[selected[:,None]+np.arange(96)])
                    future=np.asarray(values[selected[:,None]+96+np.arange(96)])
                    packed=align_pack(long[-96:],past,future,selected)
                    np.testing.assert_allclose(packed,d['packed_'+arm][row],rtol=1e-6,atol=1e-5)
        finally:store.close()
        record=dict(dataset=name,queries=len(d['sid']),audited_queries=len(ids),
                    case_pairs_physically_checked=len(ids)*50,metrics_rebuilt=True,temporal_boundaries_passed=True)
        records.append(record);print('VERIFIED MEMORY',record,flush=True)
    write_json(root/'verification.json',dict(complete=True,records=records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();verify(a.root)
