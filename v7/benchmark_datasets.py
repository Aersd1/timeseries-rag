"""Reproducible all-column benchmark on the user's eight local CSV datasets.

Run prepare with the CPU environment, then run with a validated CUDA environment.
Each stage uses a separate process, so legacy V4 imports and GPU memory are isolated.
Existing complete stages are reused only when their configuration matches exactly.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

DATASETS = {
    'ETTh1': 'ETT-small/ETTh1.csv', 'ETTh2': 'ETT-small/ETTh2.csv',
    'ETTm1': 'ETT-small/ETTm1.csv', 'ETTm2': 'ETT-small/ETTm2.csv',
    'illness': 'illness/national_illness.csv', 'weather': 'weather/weather.csv',
    'electricity': 'electricity/electricity.csv', 'traffic': 'traffic/traffic.csv',
}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def prepare(data_root, output):
    from v6.data import ingest, Windows
    base = json.loads((Path(__file__).parent/'configs/nrel.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=True)
    profiles = []
    for name, relative in DATASETS.items():
        path = (data_root/relative).resolve()
        folder = output/name
        folder.mkdir(exist_ok=True)
        frame = pd.read_csv(path)
        dates = pd.to_datetime(frame['date'], errors='raise')
        seconds = dates.diff().dt.total_seconds().dropna()
        columns = [c for c in frame.columns if c != 'date']
        values = frame[columns].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
        profile = dict(dataset=name, source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            rows=len(frame), variables=len(columns), missing=int(np.isnan(values).sum()),
            nonfinite=int((~np.isfinite(values)).sum()), duplicate_timestamps=int(dates.duplicated().sum()),
            strictly_increasing=bool(dates.is_monotonic_increasing and not dates.duplicated().any()),
            interval_seconds={str(k):int(v) for k,v in seconds.value_counts().items()},
            minus_9999=int((values == -9999).sum()), zero_memory_variance=[],
            minimum=float(np.nanmin(values)), maximum=float(np.nanmax(values)))
        if name == 'weather':
            # Preserve original files. Never interpolate held-out truth or bridge gaps.
            if not frame.loc[dates.duplicated(keep=False)].drop_duplicates().shape[0] == dates.duplicated().sum():
                raise ValueError('Unexpected conflicting weather duplicate records')
            clean = frame.drop_duplicates().copy()
            clean['date'] = pd.to_datetime(clean['date'])
            clean = clean.set_index('date')
            regular = pd.date_range(clean.index.min(),clean.index.max(),freq='10min',name='date')
            inserted = len(regular)-len(clean)
            clean = clean.reindex(regular).replace(-9999,np.nan).reset_index()
            clean_path = (folder/'weather_clean.csv').resolve()
            clean.to_csv(clean_path,index=False)
            profile['cleaning'] = dict(removed_identical_rows=len(frame)-len(frame.drop_duplicates()),
                inserted_missing_timestamps=inserted, sentinel_values_masked=profile['minus_9999'],
                clean_rows=len(clean), policy='Keep NaN gaps; exclude all windows touching any missing value; no imputation.',
                cleaned_sha256=hashlib.sha256(clean_path.read_bytes()).hexdigest())
            path = clean_path
        elif not profile['strictly_increasing'] or profile['nonfinite']:
            raise ValueError(f'{name}: resolve invalid input before benchmarking')
        elif profile['minus_9999']:
            raise ValueError(f'{name}: potential missing-value sentinel requires explicit handling')
        prior_quality = folder/'quality.json'
        if prior_quality.exists():
            old = json.loads(prior_quality.read_text(encoding='utf-8'))
            if old['sha256'] != profile['sha256'] or old.get('cleaning') != profile.get('cleaning'):
                raise ValueError(f'{name}: input changed; choose a fresh experiment directory')
        c = copy.deepcopy(base)
        c['data']['sources'] = [dict(glob=str(path), columns=columns, time_column='date',
            time_kind='datetime', group=name)]
        # Weekly illness dates need not have a uniform civil-calendar interval.
        if len(profile['interval_seconds']) == 1:
            c['data']['sources'][0]['expected_interval'] = float(seconds.iloc[0])
        if name == 'weather':
            c['data']['sources'][0]['expected_interval'] = 600
        c['training'].update(encoder_epochs=8, samples_per_series=64, batch_size=32)
        c['evaluation'].update(queries_per_series=4, k=5)
        c['moirai']['local_files_only'] = True
        c['rerank']['selection'] = 'ordinary'
        if name == 'illness':
            c.update(length=36, horizons=[24,36,48,60])
            c['training']['sample_stride'] = 4
        cp = folder/'config.json'
        if cp.exists() and json.loads(cp.read_text(encoding='utf-8')) != c:
            raise ValueError(f'Configuration changed at {cp}; choose a fresh output')
        write(cp,c)
        if not (folder/'store/catalog.json').exists():
            ingest(cp,folder/'store')
        cat = json.loads((folder/'store/catalog.json').read_text(encoding='utf-8'))
        profile['zero_memory_variance'] = [s['column'] for s in cat['series'] if s['memory_std'] <= 1e-6]
        profile['windows'] = {}
        for split in ('train','validation','test'):
            dataset = Windows(folder/'store', c, split)
            profile['windows'][split] = len(dataset)
            profile['windows'][split+'_series'] = len({sid for sid,_ in dataset.refs})
            dataset.store.close()
        from v5.common import valid_ranges
        profile['memory_windows'] = sum((b-a+st-1)//st for s in cat['series']
            for a,b,st in valid_ranges(s,c['length'],max(c['horizons']),0,s['memory_end']))
        write(folder/'quality.json',profile)
        profiles.append(profile)
        print(json.dumps(profile,ensure_ascii=True),flush=True)
    write(output/'quality.json', profiles)
    write(output/'protocol.json', dict(datasets=list(DATASETS), all_numeric_columns=True,
        seed=base['seed'], split=base['split'], standard_length=244, standard_horizons=[24,96,244],
        illness_length=36, illness_horizons=[24,36,48,60], encoder_epochs=8,
        max_train_windows_per_series=64, max_disjoint_test_queries_per_series=4,
        index_stride=1, a_top_k=5, b_candidates=20, b_top_k=5, b_selection='ordinary',
        selection_reason='Overlapping candidates permitted to obtain 20 from short memory prefixes.',
        scope='Univariate same-series retrieval; one shared A encoder per CSV; single seed; fixed training budget.',
        caveat='Exploratory controlled benchmark, not published standard splits or exhaustive hyperparameter tuning.'))


def run(output, device, names):
    status_path = output/('status_'+ '_'.join(names)+'.json')
    status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
    for name in names:
        folder = output/name
        common = ['--store', str(folder/'store')]
        cfg = ['--config', str(folder/'config.json')]
        stages = [
            ('train', ['train-a',*common,*cfg,'--output',str(folder/'train'),'--device',device], folder/'train/history.json'),
            ('index_a', ['index-a',*common,'--checkpoint',str(folder/'train/best.pt'),'--output',str(folder/'index_a'),'--device',device], folder/'index_a/manifest.json'),
            ('index_b', ['index-b',*common,*cfg,'--output',str(folder/'index_b')], folder/'index_b/manifest.json'),
            ('test', ['evaluate',*common,*cfg,'--checkpoint-a',str(folder/'train/best.pt'),
                '--index-a',str(folder/'index_a'),'--index-b',str(folder/'index_b'),
                '--output',str(folder/'test'),'--device',device], folder/'test/run.json'),
        ]
        for stage,args,marker in stages:
            key = name+'/'+stage
            if marker.exists():
                meta = json.loads(marker.read_text(encoding='utf-8'))
                complete = len(meta)==8 if stage=='train' else meta.get('complete',False)
                if complete:
                    print(f'REUSE {key}',flush=True)
                    continue
                raise ValueError(f'Incomplete stage {key}; do not silently reuse it')
            began = time.time()
            status[key] = dict(state='running', started=began)
            write(status_path,status)
            print(f'START {key}',flush=True)
            with (folder/(stage+'.log')).open('w',encoding='utf-8') as log:
                env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
                proc = subprocess.run([sys.executable,'-m','v7',*args],stdout=log,stderr=subprocess.STDOUT,env=env)
            status[key].update(state='complete' if proc.returncode==0 else 'failed', seconds=time.time()-began, exit_code=proc.returncode)
            write(status_path,status)
            print(f'END {key} {status[key]}',flush=True)
            if proc.returncode:
                raise RuntimeError(f'{key} failed; see {folder/(stage+".log")}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--data-root',type=Path,default=Path('..'))
    p.add_argument('--output',type=Path,default=Path('v7/runs/all_datasets_20261008'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--datasets',nargs='+',choices=list(DATASETS),default=list(DATASETS))
    a = p.parse_args()
    if a.action=='prepare': prepare(a.data_root,a.output)
    else: run(a.output,a.device,a.datasets)


if __name__=='__main__': main()
