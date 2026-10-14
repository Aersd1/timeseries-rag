"""Five existing routed cases plus five distinct raw-history-only cases.

No query outcome or foundation forecast participates in raw-history selection.
All ten old episodes must finish before the query history begins.
"""
import numpy as np
from .future_state_encoder import history_features, predict, decode

FAMILIES = ['dual_copy', 'dual_transport', 'dual_consensus', 'dual_trust',
    'increment_copy', 'increment_transport']


def incumbent_reference(family):
    return family=='dual_trust' or family.startswith('increment_')


class RawHistoryBank:
    def __init__(self, store, sid, end, stride=1, spacing=0):
        series = store.series[sid]
        if not 0 < end <= series['train_end'] or stride < 1:
            raise ValueError('Raw bank must be frozen no later than TRAIN end')
        self.end = int(end)
        if spacing < 0: raise ValueError('Negative raw-case separation')
        self.spacing = int(spacing)
        self.starts = np.array([p for a, b in series['runs']
            for p in range(a, min(b, end)-192+1, stride)], dtype=np.int64)
        self.past = np.stack([store.window(sid, int(p), 96) for p in self.starts]).astype(np.float64)

    def select(self, history, query_start, excluded):
        x = np.asarray(history, dtype=np.float64)
        if x.shape != (96,) or not np.isfinite(x).all():
            raise ValueError('Expected finite raw96 history')
        eligible = (self.starts+192 <= min(query_start, self.end)) & ~np.isin(self.starts, excluded)
        ids = np.flatnonzero(eligible)
        if len(ids) < 5:
            raise ValueError('Insufficient distinct fully observed history cases')
        distance = np.mean((self.past[ids]-x)**2, axis=1)
        # Absolute raw values preserve level, amplitude and trend. No detrending,
        # per-window normalization, future score or overlap penalty changes ranking.
        ranked = np.lexsort((self.starts[ids], distance))
        chosen = []
        for index in ranked:
            if all(abs(int(self.starts[ids[index]])-int(self.starts[ids[j]]))>=self.spacing for j in chosen):
                chosen.append(int(index))
                if len(chosen)==5: break
        if len(chosen)!=5: raise ValueError('Insufficient five raw cases at the required temporal separation')
        order = np.asarray(chosen)
        return self.starts[ids[order]], distance[order]


def encoded_cases(p, sid, histories, starts, selected):
    """Decode current history and align ten old paths in the same units."""
    group = p.groups[str(sid)] if p.grouped else list(range(len(p.stds)))
    local = group.index(sid)
    if p.grouped:
        from .common import sha256
        name = f'encoder_{sid}.npz'
        if sha256(p.folder/name) != p.run['group_model_hashes'][name]:
            raise ValueError('Changed group encoder')
        with np.load(p.folder/name, allow_pickle=False) as f:
            state = {k: f[k].copy() for k in f.files}
    else:
        state = p.encoder
    width = int(state.get('block_width', 16))
    hw = int(state.get('history_width', 16))
    joint = np.stack([p.joint_window(int(t), 96, group) for t in starts])
    joint[:, :, local] = histories
    def encode(values, offsets):
        features = history_features(values, offsets, p.means[group], p.stds[group],
            p.base_run['period'] or 1, weekly=p.grouped, block_width=hw)
        codes = predict(state, features)
        return np.stack([decode(code, len(group), width)[:, local] for code in codes])
    scale = np.maximum(histories.std(1), .1*p.stds[sid])
    level = histories[:, -16:].mean(1)
    encoded = encode(joint, starts)*scale[:, None]+level[:, None]
    n, k = selected.shape
    old = np.stack([p.joint_window(int(t), 96, group) for t in selected.ravel()])
    old_target = old[:, :, local].reshape(n, k, 96)
    old_scale = np.maximum(old_target.std(2), .1*p.stds[sid])
    old_level = old_target[:, :, -16:].mean(2)
    old_encoded = encode(old, selected.ravel()).reshape(n, k, 96)*old_scale[:, :, None]
    actual = np.stack([p.synchronous_values[sid][int(t)+96:int(t)+192]
        for t in selected.ravel()]).reshape(n, k, 96)
    ratio = np.clip(histories.std(1)[:, None]/old_scale, .25, 4.)
    copied = level[:, None, None]+(actual-old_level[:, :, None])*ratio[:, :, None]
    innovation = (actual-old_level[:, :, None]-old_encoded)*ratio[:, :, None]
    return encoded, copied, innovation


def fusion(encoded, copied, innovation, quantiles, history, std, distances, incumbent):
    """Equal 5+5 groups; route agreement and history support limit unsafe nudges."""
    if copied.shape[1:] != (10, 96) or innovation.shape != copied.shape:
        raise ValueError('Expected exactly ten aligned case paths')
    q = np.asarray(quantiles)
    base = q[:, 4]
    mean = innovation.mean(1)
    variance = np.mean((innovation-mean[:, None])**2, axis=1)
    confidence = mean**2/(mean**2+variance+1e-12)
    left, right = innovation[:, :5].mean(1), innovation[:, 5:].mean(1)
    agreement = np.maximum(left*right, 0)/(0.5*(left**2+right**2)+1e-12)
    # A bank outside the query's raw state is not evidence for a large correction.
    scale = np.maximum(history.std(1), .1*np.asarray(std))
    support = 1/(1+np.median(distances, axis=1)/np.maximum(scale**2, 1e-12))
    radius = np.minimum(.25*np.maximum(q[:, -1]-q[:, 0], 0), 2*scale[:, None])
    def bound(path):
        return base+np.clip(path-base, -radius, radius)
    paths = dict(dual_copy=bound(copied.mean(1)),
        dual_transport=bound(encoded+mean),
        dual_consensus=bound(encoded+confidence*agreement*support[:, None]*mean))
    # Repair the old guard's "retain incumbent forever" failure: a history-only
    # support check may attenuate an old correction; it cannot amplify its sign.
    gate = agreement*support[:, None]
    paths['dual_trust'] = bound(base+gate*(np.asarray(incumbent)-base))
    # Preserve the working five-case model. Only the difference introduced by the
    # new five cases changes its output, rather than replacing its learned state.
    within = innovation[:, :5].var(1)/5+innovation[:, 5:].var(1)/5
    difference = .5*(right-left)
    reliable = difference**2/(difference**2+.25*within+1e-12)
    transport_change = difference*reliable*np.sqrt(support[:, None])
    copy_change = .5*(copied[:, 5:].mean(1)-copied[:, :5].mean(1))*np.sqrt(support[:, None])
    paths['increment_transport'] = incumbent+np.clip(transport_change, -radius, radius)
    paths['increment_copy'] = incumbent+np.clip(copy_change, -radius, radius)
    return paths, dict(support=support, agreement=agreement.mean(1))


def choose_dual(table):
    """First two validation blocks choose once; last may only reject that choice."""
    result = {}
    for sid in np.unique(table['sid']):
        rows = np.flatnonzero(table['sid']==sid)
        rows = rows[np.argsort(table['start'][rows])]
        parts = np.array_split(rows, 3)
        default = dict(family='incumbent', beta=0., reason='no confirmed improvement')
        if min(map(len, parts)) < 3:
            result[str(int(sid))] = default
            continue
        old = table['incumbent']
        truth = table['raw_y']
        options = []
        for family in FAMILIES:
            for beta in ([1.] if family=='dual_trust' else [.1, .25, .5]):
                candidate = (old+beta*(table[family]-old)) if incumbent_reference(family) else (
                    table['moirai_direct']+beta*(table[family]-table['moirai_direct']))
                gains = []
                for ids in parts[:2]:
                    e, f = truth[ids]-old[ids], truth[ids]-candidate[ids]
                    gain = np.mean(e**2, 1)-np.mean(f**2, 1)
                    if gain.mean() <= .01*np.mean(e**2) or np.mean(gain>0)<.75 or np.mean(abs(f))>=np.mean(abs(e)):
                        break
                    gains.extend(gain.tolist())
                else:
                    options.append((-float(np.mean(gains)), family, beta))
        if not options:
            result[str(int(sid))] = default
            continue
        _, family, beta = min(options)
        ids = parts[2]
        candidate = old+beta*(table[family]-old) if incumbent_reference(family) else table['moirai_direct']+beta*(table[family]-table['moirai_direct'])
        e, f = truth[ids]-old[ids], truth[ids]-candidate[ids]
        gain = np.mean(e**2, 1)-np.mean(f**2, 1)
        ok = gain.mean()>0 and np.mean(gain>0)>=.75 and np.mean(abs(f))<np.mean(abs(e))
        result[str(int(sid))] = dict(family=family if ok else 'incumbent', beta=beta if ok else 0.,
            proposed_family=family, proposed_beta=beta, final_gain=float(gain.mean()),
            reason='5+5 confirmed' if ok else 'confirmation rejected; retain previous five-case model')
    return result


def apply_dual(table, policy):
    output = table['incumbent'].copy()
    for sid in np.unique(table['sid']):
        p = policy[str(int(sid))]
        ids = table['sid']==sid
        if p['family']=='incumbent':
            continue
        base = output[ids] if incumbent_reference(p['family']) else table['moirai_direct'][ids]
        output[ids] = base+p['beta']*(table[p['family']][ids]-base)
    return output
