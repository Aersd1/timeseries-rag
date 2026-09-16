"""Pair two completed runs, e.g. belief versus no-belief encoder ablation."""
from pathlib import Path
import pandas as pd
import numpy as np
from .common import read_json, write_json
from .analyze import rows, paired_summary


def compare(first, second, output):
    a, b = Path(first), Path(second)
    ra, rb = read_json(a/'run.json'), read_json(b/'run.json')
    for key in ('data_id','split'):
        if ra[key] != rb[key]: raise ValueError('Runs must use identical data and split')
    for key in ('seed','length','horizons','split'):
        if ra['config'][key] != rb['config'][key]: raise ValueError('Sampling/metric identity differs')
    for key in ('queries_per_series','scope','k'):
        if ra['config']['evaluation'][key] != rb['config']['evaluation'][key]: raise ValueError('Evaluation identity differs')
    if not ra['complete'] or not rb['complete']: raise ValueError('Incomplete evaluation')
    fa, fb = pd.DataFrame(rows(a/'metrics.jsonl')), pd.DataFrame(rows(b/'metrics.jsonl'))
    joined = fa.merge(fb, on=['query','sid','method','horizon'], suffixes=('', '_second'), validate='one_to_one')
    rng = np.random.default_rng(ra['config']['seed']); result=[]
    for (method,h), part in joined.groupby(['method','horizon']):
        part=part.copy(); part['persistence_nmse']=part['nmse_second']
        result.append(dict(method=method,horizon=int(h),**paired_summary(part,ra['config']['evaluation']['bootstrap_samples'],rng)))
    write_json(output, dict(interpretation='Positive improvement means FIRST has lower NMSE than SECOND. skill uses SECOND as denominator.',
                           first_checkpoint=ra['checkpoint_sha256'], second_checkpoint=rb['checkpoint_sha256'], paired=result))
    return result
