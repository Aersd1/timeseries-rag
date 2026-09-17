"""Compare V4 analog RAG (z-normalized past ED) with V6 learned RAG on the same queries.

From the repository root, conda env windmllm:

  python -m v6.compare_v4_v6 \
    --store runs/v5_nrel/store \
    --checkpoint runs/v6_nrel/encoder/best.pt \
    --index runs/v6_nrel/index \
    --vis v6/nrel_vis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from v5.common import valid_ranges
from v5.data import Store
from v6.data import Windows, scale_floor
from v6.index import diverse
from v6.inference import Retriever, analog

plt.rcParams["font.family"] = "DejaVu Sans"


def nonzero_ok(x, y):
    v = np.concatenate([x, y])
    return np.isfinite(v).all() and not np.any(v == 0) and float(np.std(x)) > 1e-7


def map_neighbor(store, c, x, sid, hit):
    m, h = c["length"], max(c["horizons"])
    values = store.window(hit["sid"], hit["start"], m + h).astype(float)
    past, future = values[:m], values[m:]
    floor = scale_floor(store.series[hit["sid"]], c)
    cs = max(float(past.std()), floor)
    mapped = float(x.mean()) + max(float(x.std()), floor) * (values - past.mean()) / cs
    return past, future, mapped


def v4_search(store, c, x, sid, start, bank):
    """V4 analog: z-normalized Euclidean distance on the 244-point history."""
    m, h = c["length"], max(c["horizons"])
    k = c["evaluation"]["k"]
    span = m + h
    starts = bank["starts"]
    allowed = np.abs(starts - int(start)) >= span
    q = np.asarray(x, dtype=np.float64)
    qz = (q - q.mean()) / max(float(q.std()), 1e-10)
    z = bank["z"][allowed]
    ss = starts[allowed]
    if not len(ss):
        return []
    d = np.mean((z - qz) ** 2, axis=1)
    cap = min(len(d), max(k * c["index"]["oversample"], k))
    pick = np.argpartition(d, cap - 1)[:cap]
    order = pick[np.argsort(d[pick], kind="stable")]
    hits = [dict(distance=float(d[i]), sid=int(sid), start=int(ss[i])) for i in order]
    return diverse(hits, k, m)


def build_v4_bank(store, c, sid):
    s = store.series[sid]
    m, h = c["length"], max(c["horizons"])
    starts = []
    for lo, end, stride in valid_ranges(s, m, h, 0, s["memory_end"], c["index"]["stride"]):
        n = (end - lo + stride - 1) // stride
        starts.append(lo + np.arange(n, dtype=np.int64) * stride)
    starts = np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)
    z = np.empty((len(starts), m), dtype=np.float32)
    for i, p in enumerate(starts):
        past = store.window(sid, int(p), m).astype(np.float64)
        z[i] = (past - past.mean()) / max(float(past.std()), 1e-10)
    return dict(starts=starts, z=z)


def plot_panel(ax, x, y, neighbors, title, cmap_name):
    m, h = len(x), len(y)
    t_hist = np.arange(-m, 0)
    t_fut = np.arange(h)
    t_all = np.arange(-m, h)
    cmap = plt.get_cmap(cmap_name)
    ylim_vals = [x, y]
    ax.plot(t_hist, x, color="tab:blue", lw=2.0, label="query history", zorder=5)
    ax.plot(t_fut, y, color="black", lw=2.0, label="true future", zorder=5)
    for i, nb in enumerate(neighbors):
        color = cmap(0.25 + 0.7 * i / max(len(neighbors) - 1, 1))
        ax.plot(t_hist, nb["past"], color=color, ls="--", lw=1.0, alpha=0.85,
                label=f"RAG{i+1} hist" if i < 3 else None)
        ax.plot(t_fut, nb["mapped"][m:], color=color, ls="-.", lw=1.1, alpha=0.9,
                label=f"RAG{i+1} future" if i < 3 else None)
        ylim_vals.extend([nb["past"], nb["mapped"][m:]])
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("t (5 min steps)")
    ax.set_ylabel("power")
    stacked = np.concatenate([np.ravel(v) for v in ylim_vals])
    lo, hi = np.nanpercentile(stacked, [1, 99])
    pad = 0.08 * (hi - lo + 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_xlim(-m, h - 1)
    return t_all


def collect_examples(retriever, n=6):
    c = retriever.c
    dataset = Windows(retriever.store.path, c, "test")
    banks = {}
    examples = []
    per_series = {}
    cap = max(1, (n + 3) // 4)
    try:
        for i in range(len(dataset)):
            batch = dataset[i]
            x, y = np.asarray(batch["x"], dtype=float), np.asarray(batch["y"], dtype=float)
            sid, start = int(batch["sid"]), int(batch["start"])
            if not nonzero_ok(x, y):
                continue
            if per_series.get(sid, 0) >= cap:
                continue
            if sid not in banks:
                banks[sid] = build_v4_bank(retriever.store, c, sid)
            v4_hits = v4_search(retriever.store, c, x, sid, start, banks[sid])
            v6 = retriever.retrieve(x, sid, start, channel="learned", leaf_budget=0)
            v4_nbs, v6_nbs = [], []
            for hit in v4_hits:
                past, future, mapped = map_neighbor(retriever.store, c, x, sid, hit)
                v4_nbs.append(dict(hit=hit, past=past, future=future, mapped=mapped))
            for hit in v6["hits"]:
                past, future, mapped = map_neighbor(retriever.store, c, x, sid, hit)
                v6_nbs.append(dict(hit=hit, past=past, future=future, mapped=mapped))
            v4_pred, _ = analog(retriever.store, c, x, scale_floor(retriever.store.series[sid], c), v4_hits)
            src = retriever.store.series[sid].get("source", f"sid={sid}")
            examples.append(dict(
                query=i, sid=sid, start=start, source=Path(str(src)).name,
                x=x, y=y, v4=v4_nbs, v6=v6_nbs,
                v4_pred=np.asarray(v4_pred, dtype=float),
                v6_pred=np.asarray(v6["prediction"], dtype=float),
                v4_starts=[int(h["start"]) for h in v4_hits],
                v6_starts=[int(h["start"]) for h in v6["hits"]],
            ))
            per_series[sid] = per_series.get(sid, 0) + 1
            if len(examples) >= n:
                break
    finally:
        dataset.store.close()
    if not examples:
        raise RuntimeError("No nonzero finite test windows found")
    return examples


def future_nmse(pred, y, x):
    scale = max(float(np.std(x)), 1e-6)
    return float(np.mean(((pred - y) / scale) ** 2))


def visualize(store, checkpoint, index, vis, device, n_examples):
    vis = Path(vis)
    vis.mkdir(parents=True, exist_ok=True)
    r = Retriever(store, checkpoint, index, device)
    try:
        examples = collect_examples(r, n_examples)
        metrics = []
        n = len(examples)
        fig, axes = plt.subplots(n, 2, figsize=(16, 3.4 * n), squeeze=False)
        for row, ex in enumerate(examples):
            qid = f"q{row+1:02d}"
            plot_panel(axes[row, 0], ex["x"], ex["y"], ex["v4"],
                       f"{qid} V4 analog RAG  sid={ex['sid']} start={ex['start']}", "Oranges")
            plot_panel(axes[row, 1], ex["x"], ex["y"], ex["v6"],
                       f"{qid} V6 learned RAG  sid={ex['sid']} start={ex['start']}", "Greens")
            single, sax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            plot_panel(sax[0], ex["x"], ex["y"], ex["v4"], f"{qid} V4 analog (z-norm ED on history)", "Oranges")
            plot_panel(sax[1], ex["x"], ex["y"], ex["v6"], f"{qid} V6 learned (future-belief embedding)", "Greens")
            single.tight_layout()
            single.savefig(vis / f"{qid}_v4_vs_v6.png", dpi=150)
            plt.close(single)
            overlap = len(set(ex["v4_starts"]) & set(ex["v6_starts"]))
            metrics.append(dict(
                query=qid, sid=ex["sid"], start=ex["start"], source=ex["source"],
                v4_starts=ex["v4_starts"], v6_starts=ex["v6_starts"],
                start_overlap=overlap,
                v4_hist_nmse=future_nmse(ex["v4_pred"], ex["y"], ex["x"]),
                v6_hist_nmse=future_nmse(ex["v6_pred"], ex["y"], ex["x"]),
            ))
        fig.tight_layout()
        fig.savefig(vis / "v4_vs_v6_grid.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(n, 2, figsize=(14, 2.8 * n), squeeze=False)
        for row, ex in enumerate(examples):
            m = len(ex["x"])
            t_hist, t_fut = np.arange(-m, 0), np.arange(len(ex["y"]))
            for col, (nbs, title, cmap_name) in enumerate((
                (ex["v4"], "V4 histories", "Oranges"),
                (ex["v6"], "V6 histories", "Greens"),
            )):
                ax = axes[row, col]
                cmap = plt.get_cmap(cmap_name)
                ax.plot(t_hist, ex["x"], color="tab:blue", lw=2.0, zorder=5, label="query")
                for i, nb in enumerate(nbs):
                    ax.plot(t_hist, nb["past"], color=cmap(0.25 + 0.7 * i / max(len(nbs) - 1, 1)),
                            ls="--", lw=1.0, alpha=0.85)
                ax.set_title(f"q{row+1:02d} {title}")
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(vis / "v4_vs_v6_history.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(n, 2, figsize=(14, 2.8 * n), squeeze=False)
        for row, ex in enumerate(examples):
            t_fut = np.arange(len(ex["y"]))
            for col, (nbs, pred, title, cmap_name) in enumerate((
                (ex["v4"], ex["v4_pred"], "V4 futures", "Oranges"),
                (ex["v6"], ex["v6_pred"], "V6 futures", "Greens"),
            )):
                ax = axes[row, col]
                cmap = plt.get_cmap(cmap_name)
                ax.plot(t_fut, ex["y"], color="black", lw=2.0, label="true future", zorder=5)
                ax.plot(t_fut, pred, color="tab:red", lw=1.6, label="analog forecast")
                for i, nb in enumerate(nbs):
                    ax.plot(t_fut, nb["mapped"][len(ex["x"]):],
                            color=cmap(0.25 + 0.7 * i / max(len(nbs) - 1, 1)),
                            ls="-.", lw=1.0, alpha=0.85)
                ax.set_title(f"q{row+1:02d} {title}")
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(vis / "v4_vs_v6_future.png", dpi=150)
        plt.close(fig)

        (vis / "v4_vs_v6_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        lines = [
            "# V4 analog vs V6 learned RAG",
            "",
            "Same NREL store, same test queries, same memory bank.",
            "V4 ranks by z-normalized Euclidean distance on the 244-point history (analog forecasting).",
            "V6 ranks by the learned future-belief embedding (query sees history only).",
            "Neighbors are greedily de-duplicated with start gap >= 244.",
            "Futures are mapped with past-only mean/std scaling.",
            "",
            "|Query|sid|start|shared starts|V4 hist-NMSE|V6 hist-NMSE|",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in metrics:
            lines.append(
                f"|{row['query']}|{row['sid']}|{row['start']}|{row['start_overlap']}"
                f"|{row['v4_hist_nmse']:.4g}|{row['v6_hist_nmse']:.4g}|"
            )
        (vis / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps(dict(examples=len(examples), vis=str(vis)), indent=2))
    finally:
        r.close()


def main():
    p = argparse.ArgumentParser(description="Plot V4 analog RAG vs V6 learned RAG")
    p.add_argument("--store", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--vis", default="v6/nrel_vis")
    p.add_argument("--device", default="cuda")
    p.add_argument("--examples", type=int, default=6)
    args = p.parse_args()
    visualize(args.store, args.checkpoint, args.index, args.vis, args.device, args.examples)


if __name__ == "__main__":
    main()
