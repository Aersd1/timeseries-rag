"""Direct PatchTST forecast vs V6 RAG analog on the same queries.

Nie et al. 2023 head: patches -> Transformer -> flatten all tokens -> Linear(pred_len).
Train on every train-split window; score MSE in query z-space
`mean(((ŷ-μ)/σ - (y-μ)/σ)²)` with `σ = max(std(x), 0.1·memory_std)`.

  python -m v6.patchtst_direct \
    --config v6/nrel.patchtst.json \
    --store runs/v5_nrel/store \
    --output runs/v6_nrel_patchtst/direct_full \
    --vis v6/nrel_vis_patchtst \
    --rag-checkpoint runs/v6_nrel_patchtst/encoder/best.pt \
    --rag-index runs/v6_nrel_patchtst/index \
    --device cuda --epochs 80
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .analyze import paired_summary
from .common import check_store, config, save_torch, write_json
from .data import Windows, scale_floor
from .model import normalize
from v5.common import sample_ranges, valid_ranges
from v5.data import Store

plt.rcParams["font.family"] = "DejaVu Sans"


class DenseWindows(torch.utils.data.Dataset):
    """All (or capped) stride windows in a split. Used only for forecast fitting."""

    def __init__(self, store_path, c, split, stride, cap_per_series=None):
        self.store = Store(store_path)
        check_store(self.store, c)
        self.c = c
        rng = np.random.default_rng(c["seed"] + {"train": 1, "validation": 2, "test": 3}[split])
        m, h = c["length"], max(c["horizons"])
        begin_end = {
            "train": lambda s: (s["memory_end"], s["train_end"]),
            "validation": lambda s: (s["train_end"], s["validation_end"]),
            "test": lambda s: (s["validation_end"], s["n"]),
        }[split]
        self.refs = []
        for s in self.store.series:
            begin, end = begin_end(s)
            ranges = list(valid_ranges(s, m, h, begin, end, stride))
            available = int(sum((b - a + st - 1) // st for a, b, st in ranges))
            count = available if cap_per_series is None else min(available, cap_per_series)
            starts = sample_ranges(ranges, count, rng)
            self.refs.extend((s["sid"], int(p)) for p in starts)
        if not self.refs:
            raise ValueError(f"No complete {split} windows")

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, i):
        sid, start = self.refs[i]
        m = self.c["length"]
        return dict(
            x=self.store.window(sid, start, m),
            y=self.store.window(sid, start + m, max(self.c["horizons"])),
            floor=np.float32(scale_floor(self.store.series[sid], self.c)),
            sid=np.int64(sid), start=np.int64(start),
        )


class PatchTSTForecast(nn.Module):
    """Univariate PatchTST with a flatten forecast head (Nie et al. 2023)."""

    def __init__(self, c):
        super().__init__()
        m = c["model"]
        length, hidden = c["length"], m["hidden"]
        self.patch_len = m.get("patch_len", 16)
        self.stride = m.get("patch_stride", 8)
        n_heads, e_layers = m.get("n_heads", 4), m.get("e_layers", 2)
        dropout = m.get("dropout", 0.1)
        if hidden % n_heads:
            raise ValueError("hidden must be divisible by n_heads")
        self.pred_len = max(c["horizons"])
        self.horizons = list(c["horizons"])
        self.n = 1 if length <= self.patch_len else 1 + math.ceil((length - self.patch_len) / self.stride)
        self.proj = nn.Linear(self.patch_len, hidden)
        self.pos = nn.Parameter(torch.randn(1, self.n, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=max(hidden * 4, 32),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.n * hidden, self.pred_len)

    def tokens(self, z):
        span = self.patch_len + (self.n - 1) * self.stride
        if z.shape[-1] < span:
            z = F.pad(z, (0, span - z.shape[-1]))
        patches = z.unfold(-1, self.patch_len, self.stride)
        if patches.shape[1] != self.n:
            raise ValueError("Patch count mismatch")
        return self.norm(self.encoder(self.proj(patches) + self.pos))

    def forward(self, x, floor):
        z, center, scale = normalize(x, floor)
        yz = self.head(self.drop(self.tokens(z).flatten(1)))
        return dict(normalized=yz, raw=yz * scale[:, None] + center[:, None], center=center, scale=scale)


def forecast_loss(model, batch):
    out = model(batch["x"], batch["floor"])
    yz = (batch["y"].float() - out["center"][:, None]) / out["scale"][:, None]
    pred = out["normalized"]
    terms = [F.mse_loss(pred[:, :h], yz[:, :h]) for h in model.horizons]
    loss = torch.stack(terms).mean()
    return loss, dict(mse=loss, **{f"h{h}": terms[i] for i, h in enumerate(model.horizons)})


def fit(store, cfg, output, device, epochs, patience=12, val_cap=512, batch_size=None):
    torch.manual_seed(cfg["seed"])
    stride = cfg["training"]["sample_stride"]
    train = DenseWindows(store, cfg, "train", stride)
    validation = DenseWindows(store, cfg, "validation", stride, cap_per_series=val_cap)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    model = PatchTSTForecast(cfg).to(device)
    batch_size = batch_size or cfg["training"]["batch_size"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-5)
    history, best, stale = [], float("inf"), 0
    print(dict(train_windows=len(train), val_windows=len(validation), batch_size=batch_size,
               hidden=cfg["model"]["hidden"], e_layers=cfg["model"].get("e_layers", 2)), flush=True)
    try:
        for epoch in range(epochs):
            model.train()
            totals, count = {}, 0
            loader = DataLoader(
                train, batch_size=batch_size, shuffle=True, num_workers=0,
                generator=torch.Generator().manual_seed(cfg["seed"] + epoch),
            )
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss, parts = forecast_loss(model, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite PatchTST forecast loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                n = len(batch["x"])
                count += n
                for k, v in parts.items():
                    totals[k] = totals.get(k, 0.0) + float(v.detach()) * n
            scheduler.step()
            model.eval()
            val_sum, val_count = 0.0, 0
            with torch.no_grad():
                for batch in DataLoader(validation, batch_size=batch_size, num_workers=0):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    val, _ = forecast_loss(model, batch)
                    val_sum += float(val) * len(batch["x"])
                    val_count += len(batch["x"])
            score = val_sum / max(val_count, 1)
            improved = score < best - 1e-4
            if improved:
                best, stale = score, 0
            else:
                stale += 1
            row = dict(epoch=epoch, validation_objective=score, lr=float(scheduler.get_last_lr()[0]),
                       stale=stale, **{k: v / count for k, v in totals.items()})
            history.append(row)
            state = dict(version=6, stage="patchtst_forecast", model=model.state_dict(), config=cfg,
                         data_id=train.store.meta["data_id"], epoch=epoch, best=best, history=history)
            save_torch(out / "last.pt", state)
            if improved:
                save_torch(out / "best.pt", state)
            write_json(out / "history.json", history)
            print(row, flush=True)
            if stale >= patience:
                print(dict(early_stop=True, best_validation=best, epoch=epoch), flush=True)
                break
    finally:
        train.store.close()
        validation.store.close()
    return dict(best_validation=best, epochs=len(history), stage="patchtst_forecast")


def load_forecast(path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("stage") != "patchtst_forecast":
        raise ValueError("Expected patchtst_forecast checkpoint")
    model = PatchTSTForecast(state["config"]).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), state


def instance_scale(x, floor):
    """Same denominator as v6.evaluate: numpy population std, memory scale floor."""
    return max(float(np.asarray(x, dtype=float).std()), float(floor))


def score_prediction(pred, y, x, floor, memory_std, horizons, query, sid, method):
    """MSE in query z-space: z = (raw - mean(x)) / max(std(x), floor)."""
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    center = float(x.mean())
    scale = instance_scale(x, floor)
    z_pred = (pred - center) / scale
    z_y = (y - center) / scale
    z_pers = (float(x[-1]) - center) / scale
    memory_scale = max(float(memory_std), 1e-6)
    rows = []
    for h in horizons:
        mse_z = float(np.mean((z_pred[:h] - z_y[:h]) ** 2))
        pers_z = float(np.mean((z_pers - z_y[:h]) ** 2))
        mse = float(np.mean((pred[:h] - y[:h]) ** 2))
        base = float(np.mean((x[-1] - y[:h]) ** 2))
        rows.append(dict(
            query=int(query), sid=int(sid), method=method, horizon=int(h),
            mse_z=mse_z, persistence_mse_z=pers_z,
            mse=mse, nmse=mse / memory_scale ** 2, history_nmse=mse_z,
            persistence_nmse=base / memory_scale ** 2,
            persistence_history_nmse=pers_z,
        ))
    return rows


def evaluate_fair(store, forecast_ckpt, rag_ckpt, rag_index, device):
    """One test loop: every method sees the same window and the same two NMSE denominators."""
    from .compare_v4_v6 import build_v4_bank, v4_search
    from .inference import Retriever, analog

    model, state = load_forecast(forecast_ckpt, device)
    c = state["config"]
    retriever = Retriever(store, rag_ckpt, rag_index, device)
    dataset = Windows(store, c, "test")
    rows, preds, banks = [], [], {}
    try:
        with torch.no_grad():
            for qi in range(len(dataset)):
                batch = dataset[qi]
                x = np.asarray(batch["x"], dtype=float)
                y = np.asarray(batch["y"], dtype=float)
                sid, start = int(batch["sid"]), int(batch["start"])
                s = dataset.store.series[sid]
                floor = scale_floor(s, c)
                xt = torch.from_numpy(np.asarray(batch["x"], dtype=np.float32)[None]).to(device)
                ft = torch.tensor([floor], dtype=torch.float32, device=device)
                direct = model(xt, ft)["raw"][0].cpu().numpy()
                vectors, prior, encoder_fc, _ = retriever.encode(np.asarray(batch["x"], dtype=np.float32), sid)
                forecasts = {
                    "persistence": np.full(len(y), float(x[-1])),
                    "patchtst_direct": np.asarray(direct, dtype=float),
                    "patchtst_encoder": np.asarray(encoder_fc, dtype=float),
                    "patchtst_prior": np.asarray(prior["mean"] * prior["scale"] + prior["center"], dtype=float),
                }
                for channel, name in (("joint", "patchtst_joint"), ("learned", "patchtst_learned")):
                    if channel in retriever.c["index"]["channels"]:
                        _, _, pred, _ = retriever.search_encoded(x, sid, start, vectors[channel], channel, 0)
                        forecasts[name] = np.asarray(pred, dtype=float)
                if sid not in banks:
                    banks[sid] = build_v4_bank(dataset.store, c, sid)
                v4_hits = v4_search(dataset.store, c, x, sid, start, banks[sid])
                v4_pred, _ = analog(dataset.store, c, x, floor, v4_hits)
                forecasts["v4_analog"] = np.asarray(v4_pred, dtype=float)
                for name, pred in forecasts.items():
                    rows.extend(score_prediction(pred, y, x, floor, s["memory_std"], c["horizons"], qi, sid, name))
                preds.append(dict(query=qi, sid=sid, start=start, x=x, y=y, floor=float(floor),
                                  memory_std=float(s["memory_std"]), forecasts=forecasts))
                print(dict(fair_query=qi + 1, total=len(dataset)), flush=True)
    finally:
        dataset.store.close()
        retriever.close()
    return pd.DataFrame(rows), preds, c


def summarize_metric(frame, value, persist, cfg, rng):
    work = frame.copy()
    work["nmse"] = work[value]
    work["persistence_nmse"] = work[persist]
    forecast, paired = [], []
    for (method, horizon), part in work.groupby(["method", "horizon"], sort=True):
        forecast.append(dict(method=method, horizon=int(horizon), metric=value,
                             **paired_summary(part, cfg["evaluation"]["bootstrap_samples"], rng)))
    for horizon, part in work.groupby("horizon"):
        methods = sorted(part["method"].unique())
        for method in methods:
            for baseline in methods:
                if method == baseline:
                    continue
                joined = part[part["method"] == method].merge(
                    part[part["method"] == baseline][["query", "nmse"]], on="query", suffixes=("", "_baseline"),
                )
                n_method = part.loc[part["method"] == method, "query"].nunique()
                if len(joined) != n_method or len(joined) < 8:
                    continue
                joined = joined.copy()
                joined["persistence_nmse"] = joined["nmse_baseline"]
                paired.append(dict(method=method, baseline=baseline, horizon=int(horizon), metric=value,
                                   **paired_summary(joined, cfg["evaluation"]["bootstrap_samples"], rng)))
    return forecast, paired


def compare_and_plot(frame, cfg, vis):
    vis = Path(vis)
    vis.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    z_f, z_p = summarize_metric(frame, "mse_z", "persistence_mse_z", cfg, rng)
    report = dict(
        note="Primary metric is MSE in query z-space: z=(raw-mean(x))/max(std(x), floor).",
        forecast_z=z_f, paired_z=z_p,
    )
    write_json(vis / "direct_vs_rag.json", report)
    frame.to_json(vis / "direct_vs_rag_scores.jsonl", orient="records", lines=True)

    def pick(table, method, horizon):
        return next((r for r in table if r["method"] == method and r["horizon"] == horizon), None)

    methods = ["persistence", "v4_analog", "patchtst_direct", "patchtst_encoder", "patchtst_prior",
               "patchtst_joint", "patchtst_learned"]
    lines = [
        "# Direct PatchTST vs RAG — z-space MSE",
        "",
        "Same 64 test queries. Every method is mapped to raw power, then scored as",
        "`mean(((ŷ-μ)/σ - (y-μ)/σ)²)` with `σ = max(std(query history), 0.1·memory_std)`.",
        "This is the same normalized space PatchTST trains in and analog mapping uses.",
        "",
        "## MSE (normalized / σ space)",
        "",
        "|H|Method|Mean MSE_z|Skill vs persistence|Win vs persistence|",
        "|---:|---|---:|---:|---:|",
    ]
    for h in cfg["horizons"]:
        for method in methods:
            r = pick(z_f, method, h)
            if r is None:
                continue
            skill = "n/a" if r["skill"] is None else f"{r['skill']:.3f}"
            lines.append(f"|{h}|{method}|{r['mean_nmse']:.5g}|{skill}|{r['win_rate']:.3f}|")
    lines += ["", "## Paired Δ MSE_z (positive = left better)", "",
              "|H|Method vs baseline|Δ MSE_z|95% CI|Win rate|",
              "|---:|---|---:|---|---:|"]
    focus = [
        ("patchtst_direct", "persistence"),
        ("patchtst_direct", "patchtst_joint"),
        ("patchtst_direct", "patchtst_learned"),
        ("patchtst_direct", "v4_analog"),
        ("patchtst_direct", "patchtst_encoder"),
        ("patchtst_joint", "persistence"),
        ("patchtst_learned", "persistence"),
    ]
    for h in cfg["horizons"]:
        for method, baseline in focus:
            r = next((p for p in z_p if p["method"] == method and p["baseline"] == baseline and p["horizon"] == h), None)
            if r is None:
                continue
            lo, hi = r["improvement_ci95"]
            lines.append(
                f"|{h}|{method} vs {baseline}|{r['paired_improvement']:.5g}|[{lo:.5g}, {hi:.5g}]|{r['win_rate']:.3f}|"
            )
    (vis / "DIRECT_VS_RAG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    styles = (
        ("persistence", "k--"),
        ("v4_analog", "x-"),
        ("patchtst_direct", "o-"),
        ("patchtst_encoder", "s-"),
        ("patchtst_joint", "^-"),
        ("patchtst_learned", "v-"),
    )
    for method, marker in styles:
        xs, ys = [], []
        for h in cfg["horizons"]:
            r = pick(z_f, method, h)
            if r is None:
                continue
            xs.append(h)
            ys.append(r["mean_nmse"])
        if xs:
            ax.plot(xs, ys, marker, label=method, lw=1.8)
    ax.set_xlabel("horizon (5 min steps)")
    ax.set_ylabel("MSE in z-space")
    ax.set_title("Direct PatchTST vs RAG (σ-normalized MSE)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(vis / "direct_vs_rag_nmse.png", dpi=150)
    plt.close(fig)
    return report


def plot_cases(preds, vis, n=6):
    from .compare_v4_v6 import nonzero_ok

    vis = Path(vis)
    chosen, per_series = [], {}
    cap = max(1, (n + 3) // 4)
    for ex in preds:
        if not nonzero_ok(ex["x"]) or per_series.get(ex["sid"], 0) >= cap:
            continue
        chosen.append(ex)
        per_series[ex["sid"]] = per_series.get(ex["sid"], 0) + 1
        if len(chosen) >= n:
            break
    if not chosen:
        chosen = preds[:n]
    fig, axes = plt.subplots(len(chosen), 1, figsize=(12, 3.1 * len(chosen)), squeeze=False)
    rows = []
    for i, ex in enumerate(chosen):
        y = np.asarray(ex["y"], dtype=float)
        fc = ex["forecasts"]
        t = np.arange(len(y))
        ax = axes[i, 0]
        ax.plot(t, y, color="black", lw=2.0, label="true future")
        ax.plot(t, fc["patchtst_direct"], color="tab:blue", lw=1.6, label="PatchTST direct")
        ax.plot(t, fc["patchtst_joint"], color="tab:purple", lw=1.6, label="RAG joint analog")
        ax.plot(t, fc["persistence"], color="0.5", ls="--", lw=1.0, label="persistence")
        ax.set_title(f"q{i+1:02d}  sid={ex['sid']} start={ex['start']}")
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylabel("power")
        scored = {}
        for name in ("patchtst_direct", "patchtst_joint", "persistence", "v4_analog"):
            scored[name] = score_prediction(
                fc[name], y, ex["x"], ex["floor"], ex["memory_std"], [len(y)], ex["query"], ex["sid"], name,
            )[0]
        rows.append(dict(
            query=f"q{i+1:02d}", sid=ex["sid"], start=ex["start"],
            direct_mse_z=scored["patchtst_direct"]["mse_z"],
            joint_mse_z=scored["patchtst_joint"]["mse_z"],
            persistence_mse_z=scored["persistence"]["mse_z"],
        ))
    axes[-1, 0].set_xlabel("t (5 min steps)")
    fig.tight_layout()
    fig.savefig(vis / "direct_vs_rag_cases.png", dpi=150)
    plt.close(fig)
    write_json(vis / "direct_vs_rag_cases.json", rows)
    return rows


def main():
    p = argparse.ArgumentParser(description="Train direct PatchTST forecast and compare with RAG")
    p.add_argument("--config", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--vis", default="v6/nrel_vis_patchtst")
    p.add_argument("--rag-checkpoint", default="runs/v6_nrel_patchtst/encoder/best.pt")
    p.add_argument("--rag-index", default="runs/v6_nrel_patchtst/index")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-cap", type=int, default=512)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--e-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()
    cfg = config(args.config)
    cfg["model"].update(backbone="patchtst", hidden=args.hidden, e_layers=args.e_layers,
                        n_heads=args.n_heads, dropout=args.dropout)
    cfg["training"]["lr"] = args.lr
    cfg["training"]["batch_size"] = args.batch_size
    out = Path(args.output)
    if not args.eval_only:
        print(fit(args.store, cfg, out, args.device, args.epochs, args.patience, args.val_cap, args.batch_size),
              flush=True)
    frame, preds, used = evaluate_fair(args.store, out / "best.pt", args.rag_checkpoint, args.rag_index, args.device)
    report = compare_and_plot(frame, used, args.vis)
    cases = plot_cases(preds, args.vis)
    table = report["forecast_z"]
    summary = {
        h: {m: next(r["mean_nmse"] for r in table if r["method"] == m and r["horizon"] == h)
            for m in ("persistence", "v4_analog", "patchtst_direct", "patchtst_encoder", "patchtst_joint", "patchtst_learned")
            if any(r["method"] == m and r["horizon"] == h for r in table)}
        for h in used["horizons"]
    }
    print(json.dumps(dict(metric="mse_z", summary=summary, cases=len(cases), vis=args.vis), indent=2))


if __name__ == "__main__":
    main()
