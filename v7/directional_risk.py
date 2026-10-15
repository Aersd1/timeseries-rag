"""Experimental risk weights for frozen RAG; never a replacement forecaster.

For N (no retrieval), R (restored 5+5) and d=R-N, squared-error gain at
weight a is 2*a*<y-N,d> - a*a*<d,d>. Only the directional inner product
needs estimating. A cluster-bootstrap ridge model estimates it from history.
"""
import numpy as np

SEEDS = (13, 41, 73, 101, 131)


def inputs(data, no_rag, restored, stats, query_only=False):
    # raw_y is deliberately absent from input construction.
    x = np.asarray(data['raw_x'], dtype=float)
    q = np.asarray(data['raw_quantiles'], dtype=float)
    sid = np.asarray(data['sid'], dtype=int)
    sd = np.asarray(stats['std'])[sid]
    level = np.asarray(stats['mean'])[sid]
    scale = np.maximum(x.std(1), np.maximum(.1*sd, 1e-6))
    center = x.mean(1)
    h = (x-center[:, None])/scale[:, None]
    n = len(x)
    def patch(v): return np.asarray(v).reshape(n, 12, 8)
    current = np.concatenate([
        patch(h).mean(2), patch(h).std(2),
        patch((q[:, 4]-center[:, None])/scale[:, None]).mean(2),
        patch((q[:, 8]-q[:, 0])/scale[:, None]).mean(2),
        patch((no_rag-q[:, 4])/scale[:, None]).mean(2),
        np.stack([(center-level)/np.maximum(sd, 1e-6),
                  np.log(scale/np.maximum(sd, 1e-6)), h[:, -1]-h[:, 0]], 1)], 1)
    past = np.asarray(data['case_past'], dtype=float)
    future = np.asarray(data['case_future'], dtype=float)
    if past.shape != (n, 10, 96) or future.shape != past.shape:
        raise ValueError('Expected the original ten history/future case pairs')
    pc = past.mean(2); ps = np.maximum(past.std(2), .1*sd[:, None])
    ps = np.maximum(ps, 1e-6)
    ph = (past-pc[:, :, None])/ps[:, :, None]
    pf = (future-pc[:, :, None])/ps[:, :, None]
    blocks = []
    for j in (0, 5):
        # Read the actual 96 old history values, including a full-resolution
        # discrepancy. Future-only matching cannot produce these features.
        hp = ph[:, j:j+5].reshape(n, 5, 12, 8)
        fp = pf[:, j:j+5].reshape(n, 5, 12, 8)
        blocks.extend([hp.mean((1, 3)), hp.mean(3).std(1),
                       fp.mean((1, 3)), fp.mean(3).std(1)])
    case = np.concatenate(blocks + [
        np.mean((ph-h[:, None])**2, 2),
        (pc-center[:, None])/scale[:, None],
        np.log(ps/scale[:, None])], 1)
    if query_only: case[:] = 0.
    delta = np.asarray(restored, dtype=float)-no_rag
    energy = np.sqrt(np.mean((delta/scale[:, None])**2, 1))
    context = np.concatenate([current, case,
                              patch(delta/scale[:, None]).mean(2), energy[:, None]], 1)
    if not np.isfinite(context).all(): raise ValueError('Nonfinite risk inputs')
    return np.clip(context, -100, 100), scale, energy


def fit(x, y, no_rag, restored, scale, starts):
    delta = (restored-no_rag)/scale[:, None]
    energy = np.sqrt(np.mean(delta**2, 1))
    target = np.mean((y-no_rag)/scale[:, None]*delta, 1)/np.maximum(energy, 1e-8)
    mean = x.mean(0); std = np.maximum(x.std(0), .25)
    a = np.c_[np.ones(len(x)), np.clip((x-mean)/std, -8, 8)]
    origins = np.unique(starts)
    if len(origins) < 3: raise ValueError('Need three completed independent origin clusters')
    coefs = []
    penalty = np.eye(a.shape[1])*10.; penalty[0, 0] = 1e-8
    for seed in SEEDS:
        sampled = np.random.default_rng(seed).choice(origins, len(origins), replace=True)
        ids = np.concatenate([np.flatnonzero(starts == t) for t in sampled])
        # Each origin has equal total weight, including across synchronous channels.
        weights = np.concatenate([np.full(np.count_nonzero(starts == t),
                                        1/np.count_nonzero(starts == t)) for t in sampled])
        weights *= len(ids)/weights.sum()
        aa = a[ids]
        coefs.append(np.linalg.solve(aa.T@(weights[:, None]*aa)+penalty,
                                     aa.T@(weights*target[ids])))
    return dict(mean=mean, std=std, coefficients=np.asarray(coefs))


def weights(model, x, energy, risk):
    a = np.c_[np.ones(len(x)), np.clip((x-model['mean'])/model['std'], -8, 8)]
    scores = a@model['coefficients'].T
    lower = scores.mean(1)-risk*scores.std(1)
    alpha = np.clip(lower/np.maximum(energy, 1e-8), 0., 1.)
    alpha[energy < 1e-8] = 0.
    return alpha


def blend(no_rag, restored, alpha):
    # Exact endpoints, including a=0, are useful for auditable refusal.
    out = np.asarray(no_rag, dtype=float).copy()
    active = alpha > 0
    out[active] += alpha[active, None]*(restored[active]-no_rag[active])
    out[alpha == 1] = restored[alpha == 1]
    return out
