"""NREL 5min 处理后数据上的 V5 测试：长零段清洗、244 点检索、历史/未来/RAG 可视化。

从仓库根目录、conda env windmllm 运行：

  python -m v5.test_nrel prepare
  python v4/build_native.py
  python -m v5 ingest --config v5/nrel.local.json --output runs/v5_nrel/store
  python -m v5 teacher --store runs/v5_nrel/store --output runs/v5_nrel/teacher
  python -m v5 train --store runs/v5_nrel/store --teacher runs/v5_nrel/teacher --output runs/v5_nrel/train --device cuda
  python -m v5 index --store runs/v5_nrel/store --checkpoint runs/v5_nrel/train/best.pt --output runs/v5_nrel/index --device cuda
  python -m v5 evaluate --store runs/v5_nrel/store --teacher runs/v5_nrel/teacher --checkpoint runs/v5_nrel/train/best.pt --index runs/v5_nrel/index --output runs/v5_nrel/test --split test --device cuda
  python -m v5 analyze --results runs/v5_nrel/test --training runs/v5_nrel/train
  python -m v5.test_nrel visualize --store runs/v5_nrel/store --checkpoint runs/v5_nrel/train/best.pt --index runs/v5_nrel/index --teacher runs/v5_nrel/teacher --vis v5/nrel_vis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import affine, norm_scale, weighted_analog
from .data import Store
from .evaluate import exact_distances
from .teacher import Labels
from .token_index import TokenLibrary
from .train import load_checkpoint

RAW_DIR = Path(
    "/ve-remi-dmz-dmz-sh-nas01/data/de_group/ghz_11401/data/wind_timeseries/NREL/raw_5min"
)
PROCESSED_DIR = Path(
    "/ve-remi-dmz-dmz-sh-nas01/data/de_group/ghz_11401/data/wind_timeseries/NREL/raw_5min_dropzero1000"
)
LENGTH = 244
MAX_ZERO_RUN = 1000


def mask_long_zeros(values: np.ndarray, max_run: int = MAX_ZERO_RUN) -> tuple[np.ndarray, dict]:
    """把连续 0 且长度 > max_run 的位置设为 NaN，短零段保留。"""
    x = np.asarray(values, dtype=float).copy()
    zero = np.isfinite(x) & (x == 0)
    if not zero.any():
        return x, dict(runs_removed=0, points_nan=0, longest_kept_zero=0)
    starts = np.flatnonzero(zero & np.r_[True, ~zero[:-1]])
    ends = np.flatnonzero(zero & np.r_[~zero[1:], True]) + 1
    removed = 0
    points = 0
    kept = 0
    for a, b in zip(starts, ends):
        n = int(b - a)
        if n > max_run:
            x[a:b] = np.nan
            removed += 1
            points += n
        else:
            kept = max(kept, n)
    return x, dict(runs_removed=removed, points_nan=points, longest_kept_zero=kept)


def compact_frame(frame: pd.DataFrame, column: str, cleaned: np.ndarray) -> pd.DataFrame:
    """丢掉长零段对应行，只在剩余有限段之间插入一行 NaN，避免 V5 把两段粘在一起。"""
    finite = np.isfinite(cleaned)
    if finite.all():
        out = frame.copy()
        out[column] = cleaned
        return out
    starts = np.flatnonzero(finite & np.r_[True, ~finite[:-1]])
    ends = np.flatnonzero(finite & np.r_[~finite[1:], True]) + 1
    parts = []
    for i, (a, b) in enumerate(zip(starts, ends)):
        chunk = frame.iloc[a:b].copy()
        chunk[column] = cleaned[a:b]
        parts.append(chunk)
        if i + 1 < len(starts):
            sep = frame.iloc[b : b + 1].copy()
            sep[column] = np.nan
            parts.append(sep)
    if not parts:
        out = frame.iloc[:0].copy()
        out[column] = []
        return out
    return pd.concat(parts, ignore_index=True)


def prepare(raw_dir: Path, out_dir: Path, column: str = "power", max_run: int = MAX_ZERO_RUN, limit_files: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"未找到 CSV：{raw_dir}")
    if limit_files > 0:
        files = files[:limit_files]
    summary = []
    for i, path in enumerate(files, 1):
        frame = pd.read_csv(path)
        if column not in frame.columns:
            raise KeyError(f"{path.name} 没有列 {column}")
        raw_power = pd.to_numeric(frame[column], errors="coerce").to_numpy()
        cleaned, stats = mask_long_zeros(raw_power, max_run)
        compact = compact_frame(frame, column, cleaned)
        dest = out_dir / path.name
        compact.to_csv(dest, index=False)
        remaining = int(np.isfinite(compact[column].to_numpy(dtype=float)).sum())
        nonzero = int(np.sum(np.isfinite(compact[column].to_numpy(dtype=float)) & (compact[column].to_numpy(dtype=float) != 0)))
        row = dict(
            file=path.name,
            **stats,
            original_rows=int(len(frame)),
            written_rows=int(len(compact)),
            remaining_finite=remaining,
            remaining_nonzero=nonzero,
        )
        summary.append(row)
        print(
            f"[{i}/{len(files)}] {path.name}: 去掉长零段 {stats['runs_removed']} 处 / {stats['points_nan']:,} 点，"
            f"写出 {len(compact):,} 行（有限 {remaining:,}，非零 {nonzero:,}）",
            flush=True,
        )
    (out_dir / "prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"处理后数据：{out_dir}")
    return summary


def site_name(source: str) -> str:
    return Path(source).name.replace("NREL_WTK_", "").replace("_2007-2013_5min.csv", "")


def is_nonzero_window(x: np.ndarray) -> bool:
    return np.isfinite(x).all() and not (x == 0).any() and float(np.std(x)) > 1e-7


def analog_from_hits(query, past, future, distances):
    if len(past) == 0:
        h = future.shape[-1] if hasattr(future, "shape") and future.ndim == 2 else LENGTH
        return np.full(h, query[-1], dtype=float)
    return weighted_analog(query, past, future, distances)


def collect_examples(store, labels, model, library, device, n_plot, analog_k, horizon):
    import torch

    m = int(model.config["length"])
    examples = []
    seen = {}
    cap = max(1, (n_plot + max(1, len(library.series)) - 1) // max(1, len(library.series)))
    model.eval()
    with torch.no_grad():
        for qi in range(len(labels)):
            if len(examples) >= n_plot:
                break
            label = labels[qi]
            sid, start = int(label["sid"]), int(label["start"])
            if sid not in library.series:
                continue
            if seen.get(sid, 0) >= cap and len(examples) + (len(library.series) - len(seen)) < n_plot:
                continue
            q = store.window(sid, start, m)
            if not is_nonzero_window(q):
                continue
            truth = store.window(sid, start + m, horizon)
            s = store.series[sid]
            floor = max(s["memory_std"] * 1e-4, 1e-6)
            scale = norm_scale(q, s["memory_std"])
            result = model(
                torch.from_numpy(q[None]).to(device),
                torch.tensor([floor], device=device),
            )
            shape = result["shape"][0].cpu().numpy()
            direct = result["prediction"][0].cpu().numpy()[:horizon]
            library.select(sid)
            budget = max(analog_k * 8, 64)
            ids = library.cosine(shape, min(library.n, budget))
            positions = np.array(library.starts[ids], dtype=np.int64)
            past = np.array([store.window(sid, int(p), m) for p in positions])
            ds = exact_distances(q, past)
            ranked = np.argsort(ds, kind="stable")[:analog_k]
            past_k = past[ranked]
            fut_k = np.array([store.window(sid, int(positions[j]) + m, horizon) for j in ranked])
            rag = analog_from_hits(q, past_k, fut_k, ds[ranked])
            mapped = affine(q, past_k, fut_k)
            persist = np.full(horizon, q[-1], dtype=float)
            examples.append(
                dict(
                    query_id=f"{sid}:{start}",
                    sid=sid,
                    start=int(start),
                    source=s["source"],
                    group=s.get("group", "NREL"),
                    query=q,
                    truth=truth,
                    token_forecaster=direct,
                    persistence=persist,
                    rag=np.asarray(rag, dtype=float),
                    rag_neighbors=[
                        dict(
                            start=int(positions[j]),
                            distance=float(np.sqrt(max(0.0, ds[j]))),
                            past=past[j],
                            future=fut_k[k],
                            analog=np.asarray(mapped[k], dtype=float),
                        )
                        for k, j in enumerate(ranked)
                    ],
                    scale=scale,
                )
            )
            seen[sid] = seen.get(sid, 0) + 1
            print(f"  可视化样本 {len(examples)}/{n_plot}  {site_name(s['source'])} start={start}", flush=True)
    if not examples:
        raise RuntimeError("测试划分里没有全部非零的 244 点查询，可放宽 is_nonzero_window 或增加 test queries")
    return examples


def plot_history_future_rag(out: Path, examples: list[dict], horizon: int):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "font.size": 11,
        }
    )
    n = len(examples)
    t_hist = np.arange(-LENGTH, 0)
    t_fut = np.arange(horizon)
    t_all = np.arange(-LENGTH, horizon)
    cmap = plt.cm.Oranges

    def neighbor_color(rank, n_hit):
        if n_hit <= 1:
            return cmap(0.85)
        return cmap(0.35 + 0.6 * (1 - (rank - 1) / (n_hit - 1)))

    def y_limits(*series):
        vals = np.concatenate([np.asarray(s, dtype=float).ravel() for s in series if s is not None and len(s)])
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            return -0.05, 1.05
        lo, hi = float(vals.min()), float(vals.max())
        pad = 0.08 * (hi - lo + 1e-6)
        return lo - pad, hi + pad

    # 总图：每条查询一行，查询 hist+future + 多条 RAG（hist+future）
    fig, axes = plt.subplots(n, 1, figsize=(15.5, 3.6 * n), squeeze=False)
    for ax, ex in zip(axes[:, 0], examples):
        q, y = ex["query"], ex["truth"]
        hits = ex["rag_neighbors"][: min(10, len(ex["rag_neighbors"]))]
        ax.plot(t_hist, q, color="#2563eb", lw=2.4, zorder=4, label="query history")
        ax.plot(t_fut, y, color="#111827", lw=2.4, zorder=4, label="query future (true)")
        ax.axvline(0, color="#94a3b8", lw=1.0, ls=":")
        extras = [q, y]
        for rank, hit in enumerate(hits, 1):
            color = neighbor_color(rank, len(hits))
            lw = 2.0 if rank == 1 else 1.15
            alpha = 0.95 if rank == 1 else 0.55
            label_h = f"RAG Top-{rank} history" if rank <= 3 else None
            label_f = f"RAG Top-{rank} future" if rank <= 3 else None
            ax.plot(t_hist, hit["past"], color=color, lw=lw, ls="--", alpha=alpha, zorder=3, label=label_h)
            ax.plot(t_fut, hit["future"], color=color, lw=lw, ls="-.", alpha=alpha, zorder=3, label=label_f)
            extras.extend([hit["past"], hit["future"]])
        ax.set_ylim(*y_limits(*extras))
        ax.set_title(
            f"{site_name(ex['source'])}  start={ex['start']}  "
            f"{len(hits)} RAG neighbors (dashed=history, dash-dot=future)",
            loc="left",
        )
        ax.set_ylabel("power")
        ax.legend(loc="upper left", ncol=4, fontsize=8)
    axes[-1, 0].set_xlabel("offset (history < 0, future >= 0)")
    fig.suptitle("V5 RAG: query vs multiple retrieved histories and futures", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "history_future_rag.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 左右分栏：左=多条历史，右=多条未来
    fig, axes = plt.subplots(n, 2, figsize=(15, 3.0 * n), squeeze=False, sharey="row")
    for r, ex in enumerate(examples):
        q, y = ex["query"], ex["truth"]
        hits = ex["rag_neighbors"][: min(10, len(ex["rag_neighbors"]))]
        axes[r, 0].plot(t_hist, q, color="#2563eb", lw=2.3, zorder=4, label="query")
        axes[r, 1].plot(t_fut, y, color="#111827", lw=2.3, zorder=4, label="query true")
        extras_h, extras_f = [q], [y]
        for rank, hit in enumerate(hits, 1):
            color = neighbor_color(rank, len(hits))
            lw = 2.0 if rank == 1 else 1.1
            alpha = 0.95 if rank == 1 else 0.5
            lab = f"Top-{rank}" if rank <= 4 or rank == len(hits) else None
            axes[r, 0].plot(t_hist, hit["past"], color=color, lw=lw, ls="--", alpha=alpha, label=lab)
            axes[r, 1].plot(t_fut, hit["future"], color=color, lw=lw, ls="-.", alpha=alpha, label=lab)
            extras_h.append(hit["past"])
            extras_f.append(hit["future"])
        axes[r, 0].axvline(0, color="#94a3b8", lw=0.8, ls=":")
        axes[r, 1].axvline(0, color="#94a3b8", lw=0.8, ls=":")
        lo, hi = y_limits(*extras_h, *extras_f)
        axes[r, 0].set_ylim(lo, hi)
        axes[r, 1].set_ylim(lo, hi)
        axes[r, 0].set_title(f"Q{r + 1} history  {site_name(ex['source'])}", loc="left", fontsize=10)
        axes[r, 1].set_title("future", loc="left", fontsize=10)
        axes[r, 0].set_ylabel("power")
        if r == 0:
            axes[r, 0].legend(fontsize=7, ncol=2, loc="upper left")
            axes[r, 1].legend(fontsize=7, ncol=2, loc="upper left")
    axes[-1, 0].set_xlabel("history offset")
    axes[-1, 1].set_xlabel("future offset")
    fig.suptitle("Multiple RAG hits: history (left) and future (right)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "history_vs_future_grid.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # 每条查询单独 Top-10：每格一条 RAG 的完整 hist+future
    for i, ex in enumerate(examples, 1):
        neighbors = ex["rag_neighbors"][: min(10, len(ex["rag_neighbors"]))]
        nplot = len(neighbors)
        fig, axes = plt.subplots(5, 2, figsize=(15, 13.5), sharex=True, sharey=False)
        q, y = ex["query"], ex["truth"]
        ctx = np.r_[q, y]
        for ax, hit, rank in zip(axes.flat, neighbors, range(1, nplot + 1)):
            retrieved = np.r_[hit["past"], hit["future"]]
            ax.plot(t_all, ctx, color="#2563eb", lw=1.8, alpha=0.95, label="query hist+true future", zorder=3)
            ax.plot(t_all, retrieved, color="#ea580c", lw=1.7, ls="--", alpha=0.95, label="RAG hist+future", zorder=4)
            ax.plot(t_all[LENGTH:], hit["analog"], color="#9a3412", lw=1.2, ls=":", alpha=0.85, label="affine future", zorder=2)
            ax.axvline(0, color="#94a3b8", lw=0.8, ls=":")
            ax.set_ylim(*y_limits(ctx, retrieved, hit["analog"]))
            ax.set_title(f"Top-{rank}  d={hit['distance']:.3f}  start={hit['start']}", loc="left", fontsize=10)
            if rank % 2 == 1:
                ax.set_ylabel("power")
            if rank >= 9:
                ax.set_xlabel("offset")
        for ax in list(axes.flat)[nplot:]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995))
        fig.suptitle(
            f"RAG Top-{nplot} history+future  ·  Q{i:02d}  {site_name(ex['source'])}  start={ex['start']}",
            fontsize=14,
            y=1.02,
        )
        fig.tight_layout(rect=(0, 0.02, 1, 0.96))
        fig.savefig(out / f"q{i:02d}_rag_top10.png", dpi=170, bbox_inches="tight")
        plt.close(fig)


def visualize(args):
    import torch

    store = Store(args.store)
    labels = Labels(args.teacher, args.split)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用")
    model, state = load_checkpoint(args.checkpoint, device)
    library = TokenLibrary(args.index)
    horizon = max(state["config"]["horizons"]) if args.horizon is None else args.horizon
    analog_k = args.analog_k if args.analog_k else int(state["config"]["evaluation"]["analog_k"])
    vis = Path(args.vis)
    print(f"测试查询 {len(labels)}，选取最多 {args.plot_queries} 条非零 244 点窗口", flush=True)
    try:
        examples = collect_examples(
            store, labels, model, library, device, args.plot_queries, analog_k, horizon
        )
        plot_history_future_rag(vis, examples, horizon)
        payload = []
        for ex in examples:
            err_rag = ex["rag"] - ex["truth"]
            err_fc = ex["token_forecaster"] - ex["truth"]
            payload.append(
                dict(
                    query_id=ex["query_id"],
                    source=ex["source"],
                    start=ex["start"],
                    rag_mae=float(np.mean(np.abs(err_rag))),
                    rag_nmse=float(np.mean(err_rag**2) / (ex["scale"] ** 2)),
                    forecast_mae=float(np.mean(np.abs(err_fc))),
                    forecast_nmse=float(np.mean(err_fc**2) / (ex["scale"] ** 2)),
                    persist_mae=float(np.mean(np.abs(ex["persistence"] - ex["truth"]))),
                    neighbor_starts=[h["start"] for h in ex["rag_neighbors"]],
                    neighbor_distances=[h["distance"] for h in ex["rag_neighbors"]],
                )
            )
        (vis / "examples_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"可视化目录：{vis.resolve()}")
        print("文件：history_future_rag.png, history_vs_future_grid.png, q##_rag_top10.png, examples_metrics.json")
    finally:
        library.close()
        store.close()


def parse_args():
    ap = argparse.ArgumentParser(description="NREL V5 长零段清洗与历史/未来/RAG 可视化")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--raw", default=str(RAW_DIR))
    p.add_argument("--output", default=str(PROCESSED_DIR))
    p.add_argument("--column", default="power")
    p.add_argument("--max-zero-run", type=int, default=MAX_ZERO_RUN)
    p.add_argument("--limit-files", type=int, default=0, help="0=全部站点")
    v = sub.add_parser("visualize")
    v.add_argument("--store", required=True)
    v.add_argument("--checkpoint", required=True)
    v.add_argument("--index", required=True)
    v.add_argument("--teacher", required=True)
    v.add_argument("--vis", default="v5/nrel_vis")
    v.add_argument("--split", choices=("validation", "test"), default="test")
    v.add_argument("--device", default="cuda")
    v.add_argument("--plot-queries", type=int, default=6)
    v.add_argument("--analog-k", type=int, default=10)
    v.add_argument("--horizon", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.command == "prepare":
        prepare(Path(args.raw), Path(args.output), args.column, args.max_zero_run, args.limit_files)
    else:
        visualize(args)


if __name__ == "__main__":
    main()
