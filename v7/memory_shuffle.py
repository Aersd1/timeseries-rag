"""Post-hoc pair-order probe; keep the entire context's numerical multiset."""
import argparse,time
from pathlib import Path
import numpy as np
from .common import sha256,write_json
from .benchmark_memory_access import read,score


def shuffle_futures(context):
    x=np.asarray(context,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=2016:raise ValueError('Expected2016-point packed contexts')
    out=x.copy();episodes=out[:,:1920].reshape(-1,10,192)
    episodes[:,:,96:]=np.roll(episodes[:,:,96:].copy(),1,axis=1)
    return out


def main():
    import torch
    from .moirai import MoiraiAdapter
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args();root=Path(a.root);summary=read(root/'summary.json');frozen=read(root/'frozen.json')
    out=root/'pair_shuffle'
    if out.exists():raise FileExistsError('Do not overwrite the supplemental probe')
    out.mkdir();names=list(frozen)
    write_json(out/'protocol.json',dict(post_hoc=True,shift=1,context_points=2016,datasets=names,
        note='Motivated by preliminary memory-access results. No value changes, fitting, or TEST-based parameter selection. Original comparisons preserved.'))
    settings=read(root/names[0]/'cohort.json')['moirai'];settings['local_files_only']=True
    model=MoiraiAdapter.pretrained(settings,'cuda');records=[]
    for name in names:
        folder=root/name;meta=read(folder/'cohort.json')
        assert sha256(folder/'cohort.json')==frozen[name]
        assert sha256(folder/'inputs.npz')==meta['inputs_sha256']
        with np.load(folder/'inputs.npz',allow_pickle=False) as f:
            real=f['packed_real'];x=shuffle_futures(real)
            np.testing.assert_array_equal(real[:,-96:],x[:,-96:])
            np.testing.assert_array_equal(real[:,:1920].reshape(-1,10,192)[:,:,:96],x[:,:1920].reshape(-1,10,192)[:,:,:96])
            np.testing.assert_array_equal(np.sort(real,axis=1),np.sort(x,axis=1))
            torch.cuda.synchronize();begin=time.perf_counter();parts=[]
            for j in range(0,len(x),a.batch_size):parts.append(model.forecast(x[j:j+a.batch_size],96)[:,4].cpu().numpy())
            torch.cuda.synchronize();elapsed=time.perf_counter()-begin;pred=np.concatenate(parts)
            m=score(f['raw_y'],pred,f['train_std'])
            np.savez_compressed(out/f'{name}.npz',prediction=pred,sid=f['sid'],start=f['start'])
        before=read(folder/'result.json')['metrics'];record=dict(dataset=name,queries=len(pred),gpu_seconds=elapsed,
            metrics=m,gain_vs_moirai96_percent=100*(1-m['mse']/before['moirai_96']['mse']),
            real_gain_vs_shuffled_percent=100*(1-before['packed_real']['mse']/m['mse']),
            input_multiset_identical=True,old_histories_and_current_history_identical=True)
        records.append(record);print('MEMORY SHUFFLE',record,flush=True)
    write_json(out/'summary.json',dict(complete=True,post_hoc=True,records=records,
        caveat='Probe of sensitivity to pair organization and continuity; not proof of causal dynamics or full-benchmark generalization'))


if __name__=='__main__':main()
