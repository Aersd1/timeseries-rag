"""CSV streaming with device chronology, missing-run boundaries, and row maps."""
from pathlib import Path
import numpy as np
import pandas as pd

EXCLUDE={'date','timestamp','time','datetime','gpu_index','index'}

def inventory(project,gpu):
    project=Path(project);gpu=Path(gpu)
    paths=[p for p in sorted(project.glob('*/*.csv')) if not p.parent.name.startswith('rag-search')]
    paths+=sorted(gpu.rglob('*.csv'))
    return [dict(path=str(p.resolve()),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,
        group='gpu' if gpu in p.parents else p.parent.name) for p in paths]

def csv_runs(file,chunk_rows=32768,max_rows=None,max_columns=None):
    counts={};carries={};row0=0;path=file['path']
    for frame in pd.read_csv(path,chunksize=chunk_rows,nrows=max_rows):
        rows=np.arange(row0,row0+len(frame),dtype=np.int64);row0+=len(frame)
        cols=[c for c in frame.columns if str(c).lower() not in EXCLUDE]
        if max_columns:cols=cols[:max_columns]
        groups=frame.groupby('gpu_index',sort=False,dropna=False).indices if 'gpu_index' in frame else {None:np.arange(len(frame))}
        for device,ids in groups.items():
            if device is not None and pd.isna(device):continue
            if device is None:dev=None
            else:
                try:dev=str(int(device)) if float(device).is_integer() else str(device)
                except (ValueError,TypeError):dev=str(device)
            base=counts.get(dev,0);counts[dev]=base+len(ids)
            part=frame.iloc[ids];partrows=rows[ids]
            for col in cols:
                raw=pd.to_numeric(part[col],errors='coerce').to_numpy(dtype=float)
                previous,oldrows=carries.get((dev,col),(np.empty(0),np.empty(0,dtype=np.int64)))
                x=np.r_[previous,raw]; rr=np.r_[oldrows,partrows];finite=np.isfinite(x)
                starts=np.flatnonzero(finite & np.r_[True,~finite[:-1]])
                ends=np.flatnonzero(finite & np.r_[~finite[1:],True])+1
                for a,b in zip(starts,ends):
                    if b-a>=244:
                        yield np.ascontiguousarray(x[a:b]),dict(source=path,column=str(col),device=dev,
                            group=file.get('group',Path(path).parent.name),start=base-len(previous)+int(a)),rr[a:b].copy()
                keep=min(243,len(x));carries[(dev,col)]=(x[-keep:].copy(),rr[-keep:].copy())

def validation_data(project,gpu):
    """Small development corpus + complete held-out-query GPU file; explicit scope."""
    files=inventory(project,gpu);parts=[];used=[]
    for file in files:
        is_origin=Path(file['path']).name=='29833367508512-r1682297-n976057.csv'
        if file['group']=='gpu' and not is_origin:continue
        limit_rows=None if is_origin else 12500
        limit_cols=None if is_origin else 3
        chunks=list(csv_runs(file,max_rows=limit_rows,max_columns=limit_cols))
        parts.extend(chunks);used.append(dict(file,max_rows=limit_rows,max_columns=limit_cols,emitted_runs=len(chunks)))
    return parts,dict(description='Development corpus: project CSVs capped at 12500 rows / 3 signal columns, plus one COMPLETE GPU CSV',
        files=used,full_requested_corpus=False)

def training_windows(parts,count=4096,seed=20260915):
    rng=np.random.default_rng(seed);eligible=[i for i,(x,_,_) in enumerate(parts) if int(len(x)*.55)>=244]
    if not eligible:raise ValueError('No sufficiently long training runs')
    windows=[]
    for _ in range(count):
        sid=int(rng.choice(eligible));x=parts[sid][0];stop=int(len(x)*.55)-244+1
        start=int(rng.integers(stop));windows.append(x[start:start+244])
    return np.array(windows)
