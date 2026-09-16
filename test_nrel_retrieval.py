"""NREL 5min 非零 244 点窗口检索测试：召回率、耗时与可视化。

只索引 power 列中连续非零（且有限）的片段；查询同样只从标准差足够大的
244 点窗口中抽样，避免全零/恒功率平台无法唯一对应来源。

环境：conda env windmllm。工作目录为 timeseries-rag。

示例：
  python test_nrel_retrieval.py
  python test_nrel_retrieval.py --limit-files 8 --queries 20 --vis nrel_vis
  python test_nrel_retrieval.py --limit-files 0   # 全部 85 个站点（建库更久）
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from cascade import Cascade
from index import IndexBuilder, SearchIndex
from model import LearnedShapeModel, znorm

LENGTH = 244
DEFAULT_DATA = (
    "/ve-remi-dmz-dmz-sh-nas01/data/de_group/ghz_11401/data/wind_timeseries/NREL/raw_5min"
)


def pick_font():
    # Droid Sans Fallback 在本机 cmap 声明含拉丁字符，实际缺字形，不能单独使用。
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("DejaVu Sans", "Noto Sans", "Liberation Sans"):
        if name in names:
            return name
    return "DejaVu Sans"


def setup_style(font: str):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )


def nrel_files(data_dir: Path, limit: int) -> list[Path]:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"未找到 CSV：{data_dir}")
    return files if limit <= 0 else files[:limit]


def nonzero_runs(values: np.ndarray):
    """按零值/NaN 切开，只保留长度 >= 244 的连续非零段。"""
    x = np.asarray(values, dtype=float)
    ok = np.isfinite(x) & (x != 0)
    starts = np.flatnonzero(ok & np.r_[True, ~ok[:-1]])
    ends = np.flatnonzero(ok & np.r_[~ok[1:], True]) + 1
    for a, b in zip(starts, ends):
        if b - a >= LENGTH:
            yield np.ascontiguousarray(x[a:b]), int(a)


def load_parts(files: list[Path], column: str):
    parts = []
    for path in files:
        frame = pd.read_csv(path, usecols=[column])
        raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        site = path.stem
        n_runs = 0
        for series, start in nonzero_runs(raw):
            meta = dict(
                source=str(path.resolve()),
                column=column,
                device=site,
                group="NREL",
                start=start,
            )
            rows = np.arange(start, start + len(series), dtype=np.int64)
            parts.append((series, meta, rows))
            n_runs += 1
        windows = sum(max(0, len(x) - LENGTH + 1) for x, _, _ in parts[-n_runs:])
        print(
            f"  {path.name}: {len(raw):,} 点, 非零段 {n_runs}, 合法 244 窗口 {windows:,}",
            flush=True,
        )
    if not parts:
        raise RuntimeError("没有任何长度 >= 244 的连续非零片段")
    return parts


def sample_training(parts, count=4096, seed=20260915):
    rng = np.random.default_rng(seed)
    eligible = [i for i, (x, _, _) in enumerate(parts) if len(x) >= LENGTH]
    windows = []
    for _ in range(count):
        sid = int(rng.choice(eligible))
        x = parts[sid][0]
        start = int(rng.integers(len(x) - LENGTH + 1))
        windows.append(x[start : start + LENGTH])
    return np.asarray(windows, dtype=float)


def sample_queries(idx: SearchIndex, n_queries: int, seed: int):
    rng = np.random.default_rng(seed)
    eligible = [
        i
        for i, info in enumerate(idx.manifest["series"])
        if info["n"] >= LENGTH + 16
    ]
    if not eligible:
        raise RuntimeError("索引中没有足够长的非零段")
    queries = []
    rejected = 0
    attempts = 0
    while len(queries) < n_queries and attempts < n_queries * 400:
        attempts += 1
        sid = int(rng.choice(eligible))
        x = np.asarray(idx.values(sid))
        start = int(rng.integers(0, len(x) - LENGTH + 1))
        q = np.array(x[start : start + LENGTH], dtype=float)
        if (q == 0).any() or not np.isfinite(q).all() or q.std() <= 1e-7:
            rejected += 1
            continue
        meta = idx.manifest["series"][sid]
        queries.append(
            dict(
                sid=sid,
                local_start=start,
                expected_start=int(meta.get("start", 0)) + start,
                source=meta.get("source"),
                column=meta.get("column"),
                device=meta.get("device"),
                values=q,
            )
        )
    if len(queries) < n_queries:
        raise RuntimeError(f"只采到 {len(queries)} 条非零非常量查询，需要 {n_queries}")
    return queries, rejected


def is_source_hit(hit: dict, query: dict) -> bool:
    return (
        hit.get("source") == query["source"]
        and hit.get("column") == query["column"]
        and hit.get("device") == query["device"]
        and int(hit["start"]) == int(query["expected_start"])
    )


def distance_recall(hits, truth_hits, k, atol=2e-5):
    if not truth_hits:
        return 0.0
    cutoff = truth_hits[min(k, len(truth_hits)) - 1]["squared_distance"]
    return sum(h["squared_distance"] <= cutoff + atol for h in hits[:k]) / k


def ensure_index(index_path: Path, parts, reuse: bool):
    manifest = index_path / "manifest.json"
    if manifest.exists():
        if not reuse:
            raise FileExistsError(f"索引已存在，改 --index 或加 --reuse-index：{index_path}")
        print(f"复用已有索引：{index_path}", flush=True)
        return json.loads(manifest.read_text(encoding="utf-8"))
    print("训练形状模型…", flush=True)
    model = LearnedShapeModel(length=LENGTH).fit(sample_training(parts))
    print("写入基础索引…", flush=True)
    builder = IndexBuilder(index_path, model)
    for i, (x, meta, rows) in enumerate(parts):
        builder.add(x, meta, rows)
        if (i + 1) % 25 == 0 or i + 1 == len(parts):
            print(
                f"  {i + 1}/{len(parts)} 段; {builder.windows:,} 窗口; {builder.points:,} 点",
                flush=True,
            )
    scope = dict(
        description="NREL WTK 5min power, contiguous nonzero runs, window=244",
        files=sorted({meta["source"] for _, meta, _ in parts}),
        column=parts[0][1]["column"],
        n_runs=len(parts),
    )
    return builder.finish(scope)


def ensure_cascade(cascade_path: Path, index_path: Path, reuse: bool):
    if (cascade_path / "manifest.json").exists():
        if not reuse:
            raise FileExistsError(f"级联索引已存在，改 --cascade-index 或加 --reuse-index：{cascade_path}")
        print(f"复用已有级联索引：{cascade_path}", flush=True)
        return json.loads((cascade_path / "manifest.json").read_text(encoding="utf-8"))
    print("构建级联索引…", flush=True)
    return Cascade.build(index_path, cascade_path)


def fill_hit_values(idx: SearchIndex, hits, length: int = LENGTH):
    for hit in hits:
        if "values" in hit and len(hit["values"]) == length:
            continue
        hit["values"] = idx.values(hit["sid"])[
            hit["local_start"] : hit["local_start"] + length
        ].tolist()
    return hits


def evaluate(engine: Cascade, queries, k: int, n_plot: int):
    idx = engine.idx
    idx.warm_metadata()
    methods = [
        ("tree exact", lambda q: idx.search(q, k=k, include_values=True)),
        (
            "cascade none",
            lambda q: engine.search(q, k=k, include_values=True, verification="none"),
        ),
        (
            "cascade scan",
            lambda q: engine.search(q, k=k, include_values=True, verification="scan"),
        ),
    ]
    # 预热，不计入常驻查询延迟
    warm = queries[0]["values"]
    idx.search(warm, k=k)
    engine.search(warm, k=k, verification="none")
    engine.search(warm, k=k, verification="scan")

    rows = []
    examples = []
    for qi, query in enumerate(queries):
        q = query["values"]
        truth = None
        for name, fn in methods:
            began = time.perf_counter()
            result = fn(q)
            wall_ms = (time.perf_counter() - began) * 1000
            if name == "tree exact":
                truth = result
            recall = distance_recall(result["hits"], truth["hits"], k)
            source_at = next(
                (r + 1 for r, h in enumerate(result["hits"]) if is_source_hit(h, query)),
                None,
            )
            row = dict(
                query_id=qi,
                method=name,
                k=k,
                ms=wall_ms,
                api_ms=result["total_ms"],
                recall=recall,
                source_hit=source_at is not None,
                source_rank=source_at,
                top1_distance=result["hits"][0]["distance"] if result["hits"] else None,
                certified=bool(result.get("certified")),
                positions_verified=int(result.get("positions_verified", 0))
                + int(result.get("verification_positions", 0)),
            )
            rows.append(row)
            if name == "cascade scan" and len(examples) < n_plot:
                fill_hit_values(idx, result["hits"][:k])
                examples.append(dict(query_id=qi, query=query, result=result))
            print(
                f"  Q{qi + 1:02d} {name:13s}  {wall_ms:7.1f} ms  "
                f"source@{source_at or '-':<3}  recall={recall:.2f}  "
                f"d={row['top1_distance']:.4g}",
                flush=True,
            )
    if not examples:
        q0 = queries[0]
        result = engine.search(q0["values"], k=k, include_values=True, verification="scan")
        fill_hit_values(idx, result["hits"][:k])
        examples.append(dict(query_id=0, query=q0, result=result))
    return rows, examples


def summarize(rows):
    summary = []
    for method in dict.fromkeys(r["method"] for r in rows):
        rr = [r for r in rows if r["method"] == method]
        ms = [r["ms"] for r in rr]
        summary.append(
            dict(
                method=method,
                queries=len(rr),
                mean_ms=float(np.mean(ms)),
                p50_ms=float(np.median(ms)),
                p95_ms=float(np.percentile(ms, 95)),
                max_ms=float(np.max(ms)),
                recall=float(np.mean([r["recall"] for r in rr])),
                source_hit_at_k=float(np.mean([r["source_hit"] for r in rr])),
                source_hit_at_1=float(
                    np.mean([r["source_rank"] == 1 for r in rr])
                ),
                certified_rate=float(np.mean([r["certified"] for r in rr])),
            )
        )
    return summary


def plot_metrics(out: Path, summary, rows, n_windows: int):
    methods = [s["method"] for s in summary]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))
    ax = axes[0]
    w = 0.36
    ax.bar(x - w / 2, [s["source_hit_at_1"] * 100 for s in summary], width=w, label="source hit@1", color="#2563eb")
    ax.bar(x + w / 2, [s["source_hit_at_k"] * 100 for s in summary], width=w, label="source hit@k", color="#ea580c")
    ax.set_xticks(x, [m.replace(" ", "\n") for m in methods])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Recall (%)")
    ax.set_title("Nonzero 244-pt source recovery", loc="left")
    ax.legend(loc="lower right")
    for i, s in enumerate(summary):
        ax.text(i - w / 2, s["source_hit_at_1"] * 100 + 1.5, f"{s['source_hit_at_1']*100:.0f}%", ha="center", fontsize=9)
        ax.text(i + w / 2, s["source_hit_at_k"] * 100 + 1.5, f"{s['source_hit_at_k']*100:.0f}%", ha="center", fontsize=9)

    ax = axes[1]
    ax.bar(x, [s["p50_ms"] for s in summary], color="#0f766e", label="median")
    ax.scatter(x, [s["p95_ms"] for s in summary], color="#111827", zorder=3, label="P95")
    ax.scatter(x, [s["max_ms"] for s in summary], color="#be123c", marker="x", zorder=3, label="max")
    ax.set_xticks(x, [m.replace(" ", "\n") for m in methods])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Resident query latency (exclude build/load)", loc="left")
    ax.legend()
    fig.suptitle(f"NREL nonzero 244-pt retrieval  ·  {n_windows:,} windows", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "recall_latency.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for method in methods:
        vals = np.sort([r["ms"] for r in rows if r["method"] == method])
        ax.plot(np.linspace(0, 100, len(vals)), vals, lw=2, label=method)
    ax.set(xlabel="Query percentile (%)", ylabel="Latency (ms)", title="Per-query latency CDF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "latency_cdf.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_retrieval(out: Path, idx: SearchIndex, example: dict, k: int):
    query = example["query"]
    result = example["result"]
    q = np.asarray(query["values"], dtype=float)
    hits = result["hits"][:k]
    if not hits:
        raise ValueError("没有可可视化的命中")
    for hit in hits:
        if "values" not in hit:
            hit["values"] = idx.values(hit["sid"])[
                hit["local_start"] : hit["local_start"] + LENGTH
            ].tolist()
    h = hits[0]
    v = np.asarray(h["values"], dtype=float)
    t = np.arange(LENGTH)
    blue, orange = "#2563eb", "#ea580c"

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1, 0.65]})
    ax[0].plot(t, q, color=blue, lw=2)
    ax[0].set_title("Query: nonzero 244-pt power window", loc="left")
    ax[0].set_ylabel("Power")
    ax[1].plot(t, q, color=blue, lw=3, alpha=0.55, label="query")
    ax[1].plot(t, v, color=orange, lw=1.8, ls="--", label="Top-1")
    ax[1].set_title(
        f"Top-1 overlay  ·  start {h['start']}  ·  z-norm ED {h['distance']:.6f}",
        loc="left",
    )
    ax[1].set_ylabel("Power")
    ax[1].legend(ncol=2)
    ax[2].plot(t, v - q, color="#059669", lw=1.6)
    ax[2].axhline(0, color="#64748b", lw=0.8)
    err = float(np.max(np.abs(v - q)))
    if err == 0:
        ax[2].set_ylim(-0.05, 0.05)
    ax[2].set_title(f"Pointwise residual (hit - query)  ·  max abs err {err:.6g}", loc="left")
    ax[2].set(xlabel="Offset within window", ylabel="Residual")
    fig.suptitle("Query vs best match", fontsize=16, y=0.99)
    fig.text(0.08, 0.008, f"file: {Path(h['source']).name}   col: {h['column']}   site: {h.get('device', '—')}", fontsize=10)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    save(fig, "query_vs_best")

    nplot = min(10, len(hits))
    fig, axes = plt.subplots(5, 2, figsize=(15, 14), sharex=True, sharey=True)
    zq = znorm(q)
    for rank, (a, hit) in enumerate(zip(axes.flat, hits[:nplot]), 1):
        a.plot(t, zq, color=blue, lw=1.8, alpha=0.8, label="query (z-norm)")
        a.plot(t, znorm(hit["values"]), color=orange, lw=1.6, ls="--", label="match (z-norm)")
        a.set_title(
            f"Top-{rank}  |  dist {hit['distance']:.3f}  |  start {hit['start']}\n{Path(hit['source']).name}",
            loc="left",
            fontsize=10,
        )
        if rank % 2 == 1:
            a.set_ylabel("z-norm")
        if rank >= 9:
            a.set_xlabel("Offset")
    for a in list(axes.flat)[nplot:]:
        a.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.966), ncol=2)
    fig.suptitle("Top-10 z-normalized shape overlay", fontsize=18, y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    save(fig, "top10_shapes")

    sid = int(h["sid"])
    start = int(h["local_start"])
    series = np.asarray(idx.values(sid))
    a = max(0, start - 1024)
    b = min(len(series), start + LENGTH + 1024)
    offset = int(idx.manifest["series"][sid].get("start", 0))
    fig, ax = plt.subplots(figsize=(13, 4.3))
    ax.plot(np.arange(a, b) + offset, series[a:b], color="#64748b", lw=1.2, label="source run (+/- 1024 pts)")
    ax.axvspan(offset + start, offset + start + LENGTH, color=orange, alpha=0.15)
    ax.plot(t + offset + start, v, color=orange, lw=2.2, label="localized 244 pts")
    ax.set_title(f"Exact location in source run: [{h['start']}, {h['start'] + LENGTH})", loc="left", fontsize=15)
    ax.set(xlabel="Sample index in site series", ylabel="Power")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    save(fig, "source_context")
    return err


def site_label(query: dict) -> str:
    return Path(query["source"]).name.replace("NREL_WTK_", "").replace("_2007-2013_5min.csv", "")


def plot_one_cascade_top10(out: Path, example: dict, k: int, stem: str):
    q = np.asarray(example["query"]["values"], dtype=float)
    hits = example["result"]["hits"][:k]
    t = np.arange(len(q))
    zq = znorm(q)
    nplot = min(10, len(hits))
    blue, orange = "#2563eb", "#ea580c"
    fig, axes = plt.subplots(5, 2, figsize=(15, 14), sharex=True, sharey=True)
    for rank, (ax, hit) in enumerate(zip(axes.flat, hits[:nplot]), 1):
        ax.plot(t, zq, color=blue, lw=1.8, alpha=0.85, label="query (z-norm)")
        ax.plot(t, znorm(hit["values"]), color=orange, lw=1.6, ls="--", label="cascade scan (z-norm)")
        src = Path(hit["source"]).name.replace("NREL_WTK_", "").replace("_2007-2013_5min.csv", "")
        ax.set_title(
            f"Top-{rank}  |  dist {hit['distance']:.3f}  |  start {hit['start']}\n{src}",
            loc="left",
            fontsize=10,
        )
        if rank % 2 == 1:
            ax.set_ylabel("z-norm")
        if rank >= 9:
            ax.set_xlabel("Offset")
    for ax in list(axes.flat)[nplot:]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.966), ncol=2)
    qid = example.get("query_id", 0) + 1
    fig.suptitle(
        f"cascade scan Top-{nplot}  ·  Q{qid:02d}  ·  {site_label(example['query'])}  start={example['query']['expected_start']}",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(out / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cascade_scan_top10(out: Path, idx: SearchIndex, examples: list[dict], k: int):
    dest = out / "cascade_scan_top10"
    dest.mkdir(parents=True, exist_ok=True)
    n = len(examples)
    if n == 0:
        raise ValueError("没有 cascade scan 结果可画")
    for i, ex in enumerate(examples, 1):
        fill_hit_values(idx, ex["result"]["hits"][:k])
        stem = f"q{ex.get('query_id', i - 1) + 1:02d}_top10"
        plot_one_cascade_top10(dest, ex, k, stem)
        print(f"  写出 {dest / (stem + '.png')}", flush=True)

    # 总览：每条查询一行，Top-1..Top-10 各一格
    ncols = min(k, 10)
    fig, axes = plt.subplots(n, ncols, figsize=(2.15 * ncols, 2.35 * n), sharex=True, sharey=True, squeeze=False)
    blue, orange = "#2563eb", "#ea580c"
    t = np.arange(LENGTH)
    for r, ex in enumerate(examples):
        q = np.asarray(ex["query"]["values"], dtype=float)
        zq = znorm(q)
        hits = ex["result"]["hits"][:ncols]
        for c in range(ncols):
            ax = axes[r, c]
            ax.plot(t, zq, color=blue, lw=1.1, alpha=0.85)
            if c < len(hits):
                ax.plot(t, znorm(hits[c]["values"]), color=orange, lw=1.0, ls="--")
                ax.set_title(f"T{c + 1} {hits[c]['distance']:.2f}", fontsize=8, pad=2)
            ax.set_xticks([])
            if c == 0:
                ax.set_ylabel(f"Q{ex.get('query_id', r) + 1:02d}", fontsize=9)
            if r == n - 1:
                ax.set_xlabel("t", fontsize=8)
    fig.suptitle("cascade scan Top-10 across queries (blue=query, orange=hit)", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    fig.savefig(dest / "overview_top10.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # 多查询：同一图里叠 Top-1..10（透明度随名次下降）
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.8 * n), squeeze=False, sharex=True)
    cmap = plt.cm.Oranges
    for ax, ex in zip(axes[:, 0], examples):
        q = np.asarray(ex["query"]["values"], dtype=float)
        ax.plot(znorm(q), color="#2563eb", lw=2.2, label="query", zorder=3)
        hits = ex["result"]["hits"][:k]
        for rank, hit in enumerate(hits, 1):
            ax.plot(
                znorm(hit["values"]),
                color=cmap(0.35 + 0.6 * (1 - (rank - 1) / max(1, len(hits) - 1))),
                lw=1.15 if rank > 1 else 1.8,
                ls="--" if rank > 1 else "-",
                alpha=0.95 if rank == 1 else 0.55,
                label=f"Top-{rank}" if rank <= 3 or rank == len(hits) else None,
            )
        ax.set_title(
            f"Q{ex.get('query_id', 0) + 1:02d}  {site_label(ex['query'])}  start={ex['query']['expected_start']}",
            loc="left",
            fontsize=11,
        )
        ax.set_ylabel("z-norm")
        ax.legend(loc="best", ncol=4, fontsize=8)
    axes[-1, 0].set_xlabel("Offset")
    fig.suptitle("cascade scan Top-10 overlay (darker = better rank)", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(dest / "overlay_top10.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return dest


def plot_examples(out: Path, idx: SearchIndex, examples: list[dict]):
    n = min(4, len(examples))
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n), squeeze=False)
    blue, orange = "#2563eb", "#ea580c"
    for ax, ex in zip(axes[:, 0], examples[:n]):
        q = np.asarray(ex["query"]["values"], dtype=float)
        hit = ex["result"]["hits"][0]
        v = np.asarray(hit["values"], dtype=float)
        ax.plot(znorm(q), color=blue, lw=2, label="query")
        ax.plot(znorm(v), color=orange, lw=1.7, ls="--", label="Top-1")
        site = Path(ex["query"]["source"]).name.replace("NREL_WTK_", "")
        ax.set_title(
            f"{site}  start={ex['query']['expected_start']}  d={hit['distance']:.4g}",
            loc="left",
        )
        ax.legend(loc="best", ncol=2)
        ax.set_ylabel("z-norm power")
    axes[-1, 0].set_xlabel("Offset")
    fig.suptitle("Top-1 shape alignment on several nonzero 244-pt queries", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "multi_query_shapes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(description="NREL 非零 244 点检索测试")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--column", default="power")
    ap.add_argument("--limit-files", type=int, default=8, help="0 表示全部站点")
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--index", default="results/nrel_index")
    ap.add_argument("--cascade-index", default="results/nrel_cascade")
    ap.add_argument("--vis", default="nrel_vis", help="可视化输出目录（timeseries-rag 下新建）")
    ap.add_argument("--plot-queries", type=int, default=6, help="cascade scan Top-10 可视化查询条数")
    ap.add_argument("--reuse-index", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data)
    index_path = Path(args.index)
    cascade_path = Path(args.cascade_index)
    vis = Path(args.vis)
    vis.mkdir(parents=True, exist_ok=True)
    font = pick_font()
    setup_style(font)

    files = nrel_files(data_dir, args.limit_files)
    print(f"数据目录：{data_dir}")
    print(f"使用 {len(files)} 个 CSV，列={args.column}，窗口={LENGTH}")
    have_index = (index_path / "manifest.json").exists()
    have_cascade = (cascade_path / "manifest.json").exists()
    reuse = args.reuse_index or have_index
    t0 = time.perf_counter()
    parts = None if have_index else load_parts(files, args.column)
    load_s = time.perf_counter() - t0
    if parts is not None:
        print(f"加载完成：{len(parts)} 个非零段，{load_s:.1f}s")
    else:
        print(f"跳过 CSV 加载，直接复用 {index_path}")

    t1 = time.perf_counter()
    manifest = ensure_index(index_path, parts, reuse)
    ensure_cascade(cascade_path, index_path, args.reuse_index or have_cascade)
    build_s = time.perf_counter() - t1
    print(f"索引窗口 {manifest['windows']:,}，建库/加载 {build_s:.1f}s")

    print("打开检索引擎…", flush=True)
    engine = Cascade(cascade_path)
    try:
        queries, rejected = sample_queries(engine.idx, args.queries, args.seed)
        print(f"抽样 {len(queries)} 条非零非常量查询（丢弃恒定候选 {rejected}）")
        print("开始检索…", flush=True)
        rows, examples = evaluate(engine, queries, args.k, args.plot_queries)
        summary = summarize(rows)
        plot_metrics(vis, summary, rows, manifest["windows"])
        err = plot_retrieval(vis, engine.idx, examples[0], args.k)
        plot_examples(vis, engine.idx, examples)
        scan_dir = plot_cascade_scan_top10(vis, engine.idx, examples, args.k)
        print(f"cascade scan Top-10 图：{scan_dir.resolve()}")
        report = dict(
            data=str(data_dir),
            files=[str(p) for p in files],
            column=args.column,
            window=LENGTH,
            windows=manifest["windows"],
            stored_points=manifest["stored_points"],
            n_runs=len(manifest["series"]),
            load_seconds=load_s,
            index_seconds=build_s,
            rejected_constant=rejected,
            font=font,
            best_max_absolute_error=err,
            summary=summary,
            rows=rows,
        )
        (vis / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"可视化目录：{vis.resolve()}")
        print("生成文件：recall_latency.png, latency_cdf.png, query_vs_best.png, top10_shapes.png, source_context.png, multi_query_shapes.png, cascade_scan_top10/, metrics.json")
    finally:
        engine.idx.close()


if __name__ == "__main__":
    main()
