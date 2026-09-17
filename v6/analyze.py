"""Paired forecast statistics, probability calibration, latency and return bundle."""
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import read_json, write_json


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def paired_summary(frame, repeats, rng):
    model, base = frame['nmse'].to_numpy(), frame['persistence_nmse'].to_numpy()
    ids = frame['sid'].unique()
    clusters = frame.assign(delta=frame.persistence_nmse-frame.nmse).groupby('sid').delta.agg(['sum','count'])
    # Resample series sums/counts; equivalent to concatenating sampled clusters.
    selections = rng.integers(0, len(clusters), size=(repeats, len(clusters)))
    boot = (clusters['sum'].to_numpy()[selections].sum(1) / clusters['count'].to_numpy()[selections].sum(1)).tolist()
    return dict(queries=len(frame), series=len(ids), mean_nmse=float(model.mean()), median_nmse=float(np.median(model)),
                skill=1-float(model.mean())/float(base.mean()) if base.mean()>0 else None,
                win_rate=float((model<base).mean()), paired_improvement=float((base-model).mean()),
                improvement_ci95=np.quantile(boot, [0.025, 0.975]).tolist() if boot else None)


def analyze(results, prior_training=None, encoder_training=None):
    out = Path(results); run = read_json(out/'run.json')
    if not run['complete']:
        raise ValueError('Evaluation is incomplete')
    frame = pd.DataFrame(rows(out/'metrics.jsonl')); timings = pd.DataFrame(rows(out/'timing.jsonl'))
    cal = pd.DataFrame(rows(out/'calibration.jsonl')); rng = np.random.default_rng(run['config']['seed'])
    summary = []
    for (method, horizon), part in frame.groupby(['method', 'horizon'], sort=True):
        summary.append(dict(method=method, horizon=int(horizon), **paired_summary(part, run['config']['evaluation']['bootstrap_samples'], rng)))
    grouped = []
    for (group, method, horizon), part in frame.groupby(['group', 'method', 'horizon'], sort=True):
        grouped.append(dict(group=group, method=method, horizon=int(horizon), queries=len(part), mean_nmse=float(part.nmse.mean())))
    paired = []
    for horizon, part in frame.groupby('horizon'):
        for method in part.method.unique():
            for baseline in ('persistence', 'history_leaves0', 'belief_leaves0', 'probabilistic_prior', 'audit_raw'):
                if method == baseline: continue
                joined = part[part.method==method].merge(part[part.method==baseline][['query','nmse']], on='query', suffixes=('', '_baseline'))
                if not len(joined): continue
                joined['persistence_nmse'] = joined['nmse_baseline']
                paired.append(dict(method=method, baseline=baseline, horizon=int(horizon),
                    **paired_summary(joined, run['config']['evaluation']['bootstrap_samples'], rng)))
    speed = []
    for method, part in timings.groupby('method'):
        speed.append(dict(method=method, queries=len(part),
            p50_ms=float(part.total_ms.quantile(.5)), p95_ms=float(part.total_ms.quantile(.95)),
            p99_ms=float(part.total_ms.quantile(.99)), max_ms=float(part.total_ms.max()),
            within100ms=float((part.total_ms<=100).mean()),
            first_query_ms=part[part.first_query].total_ms.tolist(),
            certified_fraction=float(part.certified.mean()), complete_topk_fraction=float(part.complete_topk.mean()),
            mean_returned=float(part.returned.mean()), no_match_fraction=float((part.returned==0).mean()),
            mean_scored_fraction=float((part.scored/part.eligible.clip(lower=1)).mean())))
    calibration = cal.groupby('horizon')[['crps','marginal_nll','coverage90','width90']].mean().reset_index().to_dict('records')
    audit = rows(out/'audits.jsonl')
    diagnostics = rows(out/'retrieval.jsonl')
    association, embedding_recall, similarity = {}, {}, {}
    for entry in diagnostics:
        for name, detail in entry['details'].items():
            similarity.setdefault(name,[]).append(detail)
            if 'embedding_recall_against_full' in detail:
                embedding_recall.setdefault(name,[]).append(detail['embedding_recall_against_full'])
            if detail['distance_future_spearman'] is not None:
                association.setdefault(name, []).append(detail['distance_future_spearman'])
    candidate_pairs=[]
    for name in similarity:
        if not name.startswith('joint_leaves'): continue
        baseline='learned_leaves'+name.split('leaves')[-1]
        paired_rows=[]
        for entry in diagnostics:
            left=entry['details'].get(name); right=entry['details'].get(baseline)
            if not left or not right or min(len(left['hits']),len(right['hits']))<run['config']['evaluation']['k']: continue
            if left.get('mean_history_nmse') is None or right.get('mean_history_nmse') is None: continue
            paired_rows.append(dict(query=entry['query'],sid=entry['sid'],history=left['mean_history_nmse'],
                history_baseline=right['mean_history_nmse'],nmse=left['mean_future_nmse'],persistence_nmse=right['mean_future_nmse']))
        if paired_rows:
            pairs=pd.DataFrame(paired_rows)
            candidate_pairs.append(dict(method=name,baseline=baseline,complete_pair_queries=len(pairs),
                mean_history_nmse=float(pairs.history.mean()),baseline_history_nmse=float(pairs.history_baseline.mean()),
                mean_future_nmse=float(pairs.nmse.mean()),baseline_future_nmse=float(pairs.persistence_nmse.mean()),
                future_comparison=paired_summary(pairs,run['config']['evaluation']['bootstrap_samples'],rng)))
    report = dict(forecast=summary, by_group=grouped, paired_comparisons=paired, timing=speed, calibration=calibration,
                  candidate_paired_comparisons=candidate_pairs,
                  embedding_diagnostics=run.get('embedding_diagnostics'),
                  candidate_similarity={name:dict(queries=len(items),no_match_queries=sum(not x['hits'] for x in items),
                     mean_history_nmse=float(np.mean([x['mean_history_nmse'] for x in items if x.get('mean_history_nmse') is not None])) if any(x.get('mean_history_nmse') is not None for x in items) else None,
                     max_history_nmse=max([x['max_history_nmse'] for x in items if x.get('max_history_nmse') is not None],default=None),
                     mean_future_nmse=float(np.mean([x['mean_future_nmse'] for x in items if x.get('mean_future_nmse') is not None])) if any(x.get('mean_future_nmse') is not None for x in items) else None,
                     future_nmse_by_horizon={str(h):float(np.mean([x['future_nmse_by_horizon'][str(h)] for x in items if str(h) in x.get('future_nmse_by_horizon',{})]))
                                             for h in run['config']['horizons'] if any(str(h) in x.get('future_nmse_by_horizon',{}) for x in items)}) for name,items in similarity.items()},
                  embedding_recall_against_full={k:dict(mean=float(np.mean(v)),queries=len(v)) for k,v in embedding_recall.items()},
                  audited_queries=len(audit), future_oracle_id_recall={
                      name: float(np.mean([a['future_oracle_id_recall'][name] for a in audit]))
                      for name in audit[0]['future_oracle_id_recall']} if audit else {},
                  retrieved_distance_future_spearman={k: dict(mean=float(np.mean(v)), queries=len(v)) for k,v in association.items()})
    write_json(out/'analysis.json', report)
    lines = ['# V6 results', '', f"Split: {run['split']}; queries: {run['queries']}; checkpoint epoch: {run['checkpoint_epoch']}.", '',
             '## Forecast (lower NMSE is better)', '',
             '|H|Method|N|Mean NMSE|Skill vs persistence|Win rate|', '|---:|---|---:|---:|---:|---:|']
    for row in summary:
        skill = 'undefined' if row['skill'] is None else f"{row['skill']:.3f}"
        lines.append(f"|{row['horizon']}|{row['method']}|{row['queries']}|{row['mean_nmse']:.5g}|{skill}|{row['win_rate']:.3f}|")
    lines += ['', 'NMSE uses memory-only series variance. `history_nmse` is separately retained. Every method has paired persistence values.',
              'Audit methods run on a smaller, seeded subset: compare them on matching query IDs, not against means over all queries.', '',
              '## End-to-end query latency', '', '|Method|p50 ms|p95 ms|p99 ms|max ms|≤100ms|complete top-k|scored fraction|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in speed:
        lines.append(f"|{row['method']}|{row['p50_ms']:.2f}|{row['p95_ms']:.2f}|{row['p99_ms']:.2f}|{row['max_ms']:.2f}|{row['within100ms']:.3f}|{row['complete_topk_fraction']:.3f}|{row['mean_scored_fraction']:.4f}|")
    lines += ['', '## Probability calibration', '', '|H|CRPS (normalized)|Marginal NLL|90% interval coverage|Interval width|', '|---:|---:|---:|---:|---:|']
    for row in calibration:
        lines.append(f"|{row['horizon']}|{row['crps']:.5g}|{row['marginal_nll']:.5g}|{row['coverage90']:.3f}|{row['width90']:.3f}|")
    lines += ['', '## Retrieved histories AND futures (per-candidate errors)', '',
              '|Method|Mean history NMSE|Maximum history NMSE|Mean future NMSE|Queries without matches|',
              '|---|---:|---:|---:|---:|']
    for name,row in report['candidate_similarity'].items():
        def fmt(value): return 'n/a' if value is None else f'{value:.5g}'
        lines.append(f"|{name}|{fmt(row['mean_history_nmse'])}|{fmt(row['max_history_nmse'])}|{fmt(row['mean_future_nmse'])}|{row['no_match_queries']}|")
    lines += ['', 'Candidate future errors use query-history scale and are not the same metric as memory-normalized aggregate forecast errors. Missing matches are counted separately.',
              '', '## Interpretation', '',
              '- Evidence for future-aware retrieval requires learned retrieval to beat history retrieval AND persistence on paired test queries; use the no-belief ablation too.',
              '- A calibrated 90% interval should have coverage near 0.9 without excessive width. This does not imply retrieval recall is 90%.',
              '- `future_oracle_id_recall` compares exact starts in the indexed, possibly subsampled library; ties can make this metric pessimistic.',
              '- Greedy non-overlap uses an oversampled bounded heap. Underfilled top-k is explicitly reported, never silently treated as full recall.',
              '- `certified` concerns stored embedding distances only. Budgeted early stopping is approximate. True-future similarity is not guaranteed.',
              '- Bootstrap intervals cluster by series; few series give weak uncertainty estimates. Cross-series calendar alignment is not verified.',
              '- No 10B-point or <100ms claim follows from a small run. Timings exclude process/model/index object loading; first query is retained separately.',
              '- Checkpoint epoch -1 means an untrained smoke fixture, NOT an experiment.', '', '![Overview](overview.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    sf = pd.DataFrame(summary)
    for method, part in sf.groupby('method'):
        axes[0].plot(part.horizon, part.mean_nmse, marker='o', label=method)
    axes[0].set(xlabel='Horizon', ylabel='Memory-normalized MSE', yscale='symlog')
    axes[0].legend(fontsize=6)
    axes[1].barh([s['method'] for s in speed], [s['p95_ms'] for s in speed])
    axes[1].axvline(100, color='red', linestyle='--'); axes[1].set_xlabel('p95 end-to-end query ms')
    axes[1].tick_params(axis='y', labelsize=7)
    axes[2].plot([r['horizon'] for r in calibration], [r['coverage90'] for r in calibration], marker='o')
    axes[2].axhline(.9, color='red', linestyle='--'); axes[2].set(xlabel='Horizon', ylabel='90% marginal interval coverage', ylim=(0,1))
    fig.tight_layout(); fig.savefig(out/'overview.png', dpi=160); plt.close(fig)
    for ex in read_json(out/'examples_private.json'):
        fig, ax = plt.subplots(figsize=(12, 4)); m = len(ex['x']); h = len(ex['y'])
        ax.plot(np.arange(-m, 0), ex['x'], label='Observed history', color='black')
        ax.plot(np.arange(h), ex['y'], label='True future', color='black', linestyle='--')
        plotted=['persistence','probabilistic_prior']
        for channel in ('learned','history','belief','joint'):
            name=next((name for name in ex['forecasts'] if name.startswith(channel+'_leaves')),None)
            if name: plotted.append(name)
        for name in plotted:
            if name in ex['forecasts']:
                ax.plot(np.arange(h), ex['forecasts'][name], label=name, alpha=.8)
        ax.fill_between(np.arange(h), ex['lower90'], ex['upper90'], alpha=.15, label='Prior 90% marginal interval')
        ax.axvline(0, color='gray'); ax.legend(fontsize=7); ax.set_title(f"query {ex['query']}, series {ex['sid']}")
        fig.tight_layout(); fig.savefig(out/f"example_private_{ex['query']:04d}.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(np.arange(-m,h),ex['x']+ex['y'],color='black',label='Query history + held-out future')
        for match in ex.get('matches',[]):
            ax.plot(np.arange(-m,h),match['mapped'],alpha=.75,label=f"retrieved sid={match['sid']} start={match['start']}")
        ax.axvline(0,color='gray',linestyle='--'); ax.legend(fontsize=7)
        ax.set_title('Retrieved history and continuation; scaled using past only')
        fig.tight_layout(); fig.savefig(out/f"retrieved_private_{ex['query']:04d}.png",dpi=150); plt.close(fig)
    public = dict(run); public['config'] = dict(run['config']); public['config'].pop('data', None)
    write_json(out/'run_public.json', public)
    extras = []
    for label, source in [('prior_history', prior_training), ('encoder_history', encoder_training)]:
        if source:
            name = label+'.json'; write_json(out/name, read_json(Path(source)/'history.json')); extras.append(name)
    names = ['REPORT.md','analysis.json','overview.png','run_public.json','metrics.jsonl','timing.jsonl','calibration.jsonl','audits.jsonl','retrieval.jsonl']+extras
    if (out/'fusion_stats.json').exists(): names.append('fusion_stats.json')
    with zipfile.ZipFile(out/'analysis_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name in names:
            bundle.write(out/name, name)
    return report
