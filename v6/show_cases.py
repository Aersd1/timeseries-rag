"""Case studies: V4 analog vs V6 learned vs V6 joint on the same test queries.

  python -m v6.show_cases \
    --store runs/v5_nrel/store \
    --checkpoint runs/v6_nrel_joint/encoder/best.pt \
    --index runs/v6_nrel_joint/index \
    --vis v6/nrel_vis_joint
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from v6.compare_v4_v6 import build_v4_bank, map_neighbor, nonzero_ok, v4_search
from v6.data import Windows, scale_floor
from v6.inference import Retriever, analog

plt.rcParams["font.family"] = "DejaVu Sans"


def history_nmse(query, past, floor):
    q = np.asarray(query, dtype=float)
    p = np.asarray(past, dtype=float)
    qz = (q - q.mean()) / max(float(q.std()), floor)
    pz = (p - p.mean()) / max(float(p.std()), floor)
    return float(np.mean((qz - pz) ** 2))


def future_nmse(pred, y, x):
    scale = max(float(np.std(x)), 1e-6)
    return float(np.mean(((np.asarray(pred) - y) / scale) ** 2))


def attach_errors(nbs, x, y, sid, store, c):
    floor = scale_floor(store.series[sid], c)
    out = []
    for nb in nbs:
        mapped = nb["mapped"]
        out.append(dict(
            **nb,
            history_nmse=history_nmse(x, nb["past"], floor),
            future_nmse=future_nmse(mapped[len(x):], y, x),
        ))
    return out


def plot_panel(ax, x, y, neighbors, title, cmap_name):
    m, h = len(x), len(y)
    t_hist, t_fut = np.arange(-m, 0), np.arange(h)
    cmap = plt.get_cmap(cmap_name)
    vals = [x, y]
    ax.plot(t_hist, x, color="tab:blue", lw=2.0, label="query history", zorder=5)
    ax.plot(t_fut, y, color="black", lw=2.0, label="true future", zorder=5)
    for i, nb in enumerate(neighbors):
        color = cmap(0.25 + 0.7 * i / max(len(neighbors) - 1, 1))
        ax.plot(t_hist, nb["past"], color=color, ls="--", lw=1.0, alpha=0.85,
                label=f"RAG{i+1} hist" if i < 3 else None)
        ax.plot(t_fut, nb["mapped"][m:], color=color, ls="-.", lw=1.1, alpha=0.9,
                label=f"RAG{i+1} future" if i < 3 else None)
        vals.extend([nb["past"], nb["mapped"][m:]])
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("t (5 min steps)")
    ax.set_ylabel("power")
    stacked = np.concatenate([np.ravel(v) for v in vals])
    lo, hi = np.nanpercentile(stacked, [1, 99])
    pad = 0.08 * (hi - lo + 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_xlim(-m, h - 1)


def collect(retriever, n=6):
    c = retriever.c
    dataset = Windows(retriever.store.path, c, "test")
    banks, examples, per_series = {}, [], {}
    cap = max(1, (n + 3) // 4)
    channels = [ch for ch in ("learned", "joint") if ch in c["index"]["channels"]]
    try:
        for i in range(len(dataset)):
            batch = dataset[i]
            x, y = np.asarray(batch["x"], dtype=float), np.asarray(batch["y"], dtype=float)
            sid, start = int(batch["sid"]), int(batch["start"])
            if not nonzero_ok(x) or per_series.get(sid, 0) >= cap:
                continue
            if sid not in banks:
                banks[sid] = build_v4_bank(retriever.store, c, sid)
            v4_hits = v4_search(retriever.store, c, x, sid, start, banks[sid])
            v4_pred, _ = analog(retriever.store, c, x, scale_floor(retriever.store.series[sid], c), v4_hits)
            methods = {
                "v4": dict(
                    hits=v4_hits,
                    nbs=attach_errors(
                        [dict(hit=h, **dict(zip(("past", "future", "mapped"), map_neighbor(retriever.store, c, x, sid, h))))
                         for h in v4_hits],
                        x, y, sid, retriever.store, c,
                    ),
                    pred=np.asarray(v4_pred, dtype=float),
                )
            }
            for ch in channels:
                result = retriever.retrieve(x, sid, start, channel=ch, leaf_budget=0)
                nbs = []
                for hit in result["hits"]:
                    past, future, mapped = map_neighbor(retriever.store, c, x, sid, hit)
                    nbs.append(dict(hit=hit, past=past, future=future, mapped=mapped))
                methods[ch] = dict(
                    hits=result["hits"],
                    nbs=attach_errors(nbs, x, y, sid, retriever.store, c),
                    pred=np.asarray(result["prediction"], dtype=float),
                    stats=result["stats"],
                )
            src = Path(str(retriever.store.series[sid].get("source", f"sid={sid}"))).name
            examples.append(dict(query=i, sid=sid, start=start, source=src, x=x, y=y, methods=methods))
            per_series[sid] = per_series.get(sid, 0) + 1
            if len(examples) >= n:
                break
    finally:
        dataset.store.close()
    if not examples:
        raise RuntimeError("No finite nonzero-history test windows found")
    return examples, channels


def visualize(store, checkpoint, index, vis, device, n_examples):
    vis = Path(vis)
    vis.mkdir(parents=True, exist_ok=True)
    r = Retriever(store, checkpoint, index, device)
    try:
        examples, channels = collect(r, n_examples)
        backbone = str(r.c.get("model", {}).get("backbone", "cnn"))
        tag = "PatchTST " if backbone == "patchtst" else ""
        method_order = [("v4", "V4 analog", "Oranges")]
        if "learned" in channels:
            method_order.append(("learned", f"V6 {tag}learned", "Greens"))
        if "joint" in channels:
            method_order.append(("joint", f"V6 {tag}joint", "Purples"))
        rows, cols = len(examples), len(method_order)
        grid, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 3.5 * rows), squeeze=False)
        metrics = []
        for i, ex in enumerate(examples):
            qid = f"q{i+1:02d}"
            case_fig, case_ax = plt.subplots(cols, 1, figsize=(12, 3.6 * cols), sharex=True, squeeze=False)
            row = dict(query=qid, sid=ex["sid"], start=ex["start"], source=ex["source"])
            for j, (key, title, cmap) in enumerate(method_order):
                method = ex["methods"][key]
                heading = f"{qid} {title}  sid={ex['sid']} start={ex['start']}  n={len(method['nbs'])}"
                plot_panel(axes[i, j], ex["x"], ex["y"], method["nbs"], heading, cmap)
                plot_panel(case_ax[j, 0], ex["x"], ex["y"], method["nbs"], heading, cmap)
                row[f"{key}_starts"] = [int(nb["hit"]["start"]) for nb in method["nbs"]]
                row[f"{key}_n"] = len(method["nbs"])
                row[f"{key}_forecast_nmse"] = future_nmse(method["pred"], ex["y"], ex["x"])
                row[f"{key}_mean_hist_nmse"] = float(np.mean([nb["history_nmse"] for nb in method["nbs"]])) if method["nbs"] else None
                row[f"{key}_mean_fut_nmse"] = float(np.mean([nb["future_nmse"] for nb in method["nbs"]])) if method["nbs"] else None
            v4_starts = set(row.get("v4_starts", []))
            for key, _, _ in method_order[1:]:
                row[f"v4_{key}_overlap"] = len(v4_starts & set(row.get(f"{key}_starts", [])))
            metrics.append(row)
            case_fig.tight_layout()
            case_fig.savefig(vis / f"{qid}_case.png", dpi=150)
            plt.close(case_fig)
        grid.tight_layout()
        grid.savefig(vis / "cases_grid.png", dpi=140)
        plt.close(grid)

        hist_fig, hist_ax = plt.subplots(rows, cols, figsize=(6.4 * cols, 2.6 * rows), squeeze=False)
        fut_fig, fut_ax = plt.subplots(rows, cols, figsize=(6.4 * cols, 2.6 * rows), squeeze=False)
        for i, ex in enumerate(examples):
            m = len(ex["x"])
            t_hist, t_fut = np.arange(-m, 0), np.arange(len(ex["y"]))
            for j, (key, title, cmap_name) in enumerate(method_order):
                cmap = plt.get_cmap(cmap_name)
                nbs = ex["methods"][key]["nbs"]
                hist_ax[i, j].plot(t_hist, ex["x"], color="tab:blue", lw=2.0, zorder=5, label="query")
                fut_ax[i, j].plot(t_fut, ex["y"], color="black", lw=2.0, zorder=5, label="true future")
                fut_ax[i, j].plot(t_fut, ex["methods"][key]["pred"], color="tab:red", lw=1.5, label="analog forecast")
                for k, nb in enumerate(nbs):
                    color = cmap(0.25 + 0.7 * k / max(len(nbs) - 1, 1))
                    hist_ax[i, j].plot(t_hist, nb["past"], color=color, ls="--", lw=1.0, alpha=0.85)
                    fut_ax[i, j].plot(t_fut, nb["mapped"][m:], color=color, ls="-.", lw=1.0, alpha=0.85)
                hist_ax[i, j].set_title(f"q{i+1:02d} {title} history")
                fut_ax[i, j].set_title(f"q{i+1:02d} {title} future")
                hist_ax[i, j].legend(fontsize=7)
                fut_ax[i, j].legend(fontsize=7)
        hist_fig.tight_layout(); hist_fig.savefig(vis / "cases_history.png", dpi=140); plt.close(hist_fig)
        fut_fig.tight_layout(); fut_fig.savefig(vis / "cases_future.png", dpi=140); plt.close(fut_fig)

        (vis / "cases_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        headers = ["Query", "sid", "start"]
        for key, title, _ in method_order:
            headers += [f"{title} n", f"{title} histNMSE", f"{title} futNMSE", f"{title} forecast"]
        lines = [
            "# V6 joint retrieval case studies",
            "",
            "Same NREL store and test queries. History-only example filter; not a benchmark.",
            "V4: z-normalized Euclidean distance on 244-point history.",
            "V6 learned: future-belief embedding.",
            "V6 joint: concatenated learned+history vectors plus raw-history NMSE gate (tau=0.5).",
            "Candidate future NMSE uses query-history scale after past-only mean/std mapping.",
            "",
            "|" + "|".join(headers) + "|",
            "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
        ]
        for row in metrics:
            cells = [row["query"], str(row["sid"]), str(row["start"])]
            for key, _, _ in method_order:
                cells += [
                    str(row[f"{key}_n"]),
                    "n/a" if row[f"{key}_mean_hist_nmse"] is None else f"{row[f'{key}_mean_hist_nmse']:.4g}",
                    "n/a" if row[f"{key}_mean_fut_nmse"] is None else f"{row[f'{key}_mean_fut_nmse']:.4g}",
                    f"{row[f'{key}_forecast_nmse']:.4g}",
                ]
            lines.append("|" + "|".join(cells) + "|")
        (vis / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps(dict(examples=len(examples), channels=channels, vis=str(vis)), indent=2))
    finally:
        r.close()


def main():
    p = argparse.ArgumentParser(description="Plot V4 / V6 learned / V6 joint retrieval cases")
    p.add_argument("--store", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--vis", default="v6/nrel_vis_joint")
    p.add_argument("--device", default="cuda")
    p.add_argument("--examples", type=int, default=6)
    args = p.parse_args()
    visualize(args.store, args.checkpoint, args.index, args.vis, args.device, args.examples)


if __name__ == "__main__":
    main()
