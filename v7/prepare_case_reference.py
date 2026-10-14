"""Train the adapter on the remaining error of the ACTUAL frozen five-case model."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import sha256,write_json
from .query_guarded_bank import GuardedBankPredictor
from .benchmark_error_correction import load


def prepare(source,out,split):
    old_meta=json.loads((source/'preparation.json').read_text());dual=Path(old_meta['source'])
    cfg=json.loads((dual/'contract.json').read_text());previous=GuardedBankPredictor(old_meta['store'],cfg['source'],adapter=False)
    try:
        with np.load(source/f'{split}.npz',allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
        p=previous.original;cache=load(p.cache,split);probes=Path(p.base_run['source_probes'])
        manifest=json.loads((probes/'manifest.json').read_text())
        if sha256(probes/'manifest.json')!=p.base_run['source_probe_manifest_sha256'] or sha256(probes/f'{split}.npz')!=manifest['files'][split]:raise ValueError('Changed historical probe')
        with np.load(probes/f'{split}.npz',allow_pickle=False) as f:
            for k in ['sid','start']:np.testing.assert_array_equal(f[k],data[k])
            probe=f['quantiles'].copy()
        reference=np.zeros_like(data['raw_x'],dtype=np.float64);no_reference=np.zeros_like(reference)
        for i,(sid,start) in enumerate(zip(data['sid'],data['start'])):
            m={k:cache[k][i] for k in ['weights','selected_starts','candidate_starts','history_scores','future_scores']}
            result=previous.predict_cached(data['raw_x'][i],int(sid),int(start),data['raw_quantiles'][i:i+1],probe[i:i+1],
                training=split=='train',metadata=m,validation=split=='validation')
            np.testing.assert_array_equal(result['selected_starts'],data['selected_starts'][i])
            reference[i]=result['forecasts']['guarded_rag'];no_reference[i]=result['forecasts']['guarded_no_rag']
            if (i+1)%1000==0:print('REFERENCE',out.name,split,i+1,'/',len(reference),flush=True)
        data.update(reference=reference,no_reference=no_reference,probe_quantiles=probe)
        path=out/f'{split}.npz';np.savez_compressed(path,**data)
        return dict(queries=len(reference),case_pairs=len(reference)*5,sha256=sha256(path))
    finally:previous.close()


def initial(source,out):
    out.mkdir(parents=True,exist_ok=False);meta=json.loads((source/'preparation.json').read_text());stats=json.loads((source/'stats.json').read_text())
    stats.update(residual_reference=True);write_json(out/'stats.json',stats)
    meta.update(residual_reference=True,reference_source=str(source),stats_sha256=sha256(out/'stats.json'),
        files={split:prepare(source,out,split) for split in ['train','validation']},
        training='Frozen five-case TRAIN prediction as residual reference; in-sample source model, not OOF. No reference uses the current query true future; all retrieval cases end before query history starts.')
    write_json(out/'preparation.json',meta);print('PREPARED RESIDUAL FIVE',out.name,flush=True)


def copy_feature(source,out,split):
    expected=json.loads((source/'run.json').read_text())['test_info']['sha256'] if split=='test' else json.loads((source/'preparation.json').read_text())['files'][split]['sha256']
    if sha256(source/f'{split}.npz')!=expected:raise ValueError('Changed residual reference feature cache')
    if split=='test' and not (out/'policy.json').exists():raise ValueError('Freeze policy before TEST copy')
    import shutil
    shutil.copyfile(source/f'{split}.npz',out/f'{split}.npz')
    meta=json.loads((source/'preparation.json').read_text())
    n=json.loads((source/'run.json').read_text())['queries'] if split=='test' else meta['files'][split]['queries']
    return dict(queries=n,case_pairs=n*5,sha256=sha256(out/f'{split}.npz'))


def paired_initial(source,out):
    out.mkdir(parents=True,exist_ok=False);meta=json.loads((source/'preparation.json').read_text());stats=json.loads((source/'stats.json').read_text())
    stats.update(paired_attention=True,predictive_contrastive=True,metadata_width=13,
        contrastive_weight=.05,score_temperature=.1)
    write_json(out/'stats.json',stats)
    meta.update(reference_feature_source=str(source),reference_feature_preparation_sha256=sha256(source/'preparation.json'),
        paired_attention=True,predictive_contrastive=True,stats_sha256=sha256(out/'stats.json'),
        files={split:copy_feature(source,out,split) for split in ['train','validation']})
    write_json(out/'preparation.json',meta);print('PREPARED PAIRED PREDICTIVE FIVE',out.name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--paired',action='store_true');a=p.parse_args()
    for name in a.datasets:(paired_initial if a.paired else initial)(Path(a.source)/name,Path(a.output)/name)
