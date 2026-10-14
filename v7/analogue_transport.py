"""Transfer observed analogue innovations, preserving the query state prediction."""
import numpy as np
from .safe_error_policy import bound_delta


def transport_paths(encoded,cases,old_encoded):
    encoded=np.asarray(encoded,dtype=float);cases=np.asarray(cases,dtype=float);old_encoded=np.asarray(old_encoded,dtype=float)
    if encoded.shape!=(96,) or cases.shape!=(5,96) or old_encoded.shape!=(5,96):raise ValueError('Expected query and five aligned96-point case paths')
    if not np.isfinite(encoded).all() or not np.isfinite(cases).all() or not np.isfinite(old_encoded).all():raise ValueError('Nonfinite transport inputs')
    return encoded[None]+cases-old_encoded


def experts(components,weights):
    paths=transport_paths(components['encoded_path'],components['cases'],components['old_encoded_paths'])
    w=np.asarray(weights,dtype=float);w=w/w.sum();innovation=w@(paths-components['encoded_path'][None])
    variance=w@((paths-components['encoded_path'][None]-innovation)**2)
    confidence=(innovation**2)/(innovation**2+variance+1e-12)
    raw=components['encoded_path']+innovation
    # Consensus only discounts the transferred innovation, not the query forecast.
    consensus=components['encoded_path']+confidence*innovation
    result=dict(components['deltas'])
    for name,path in [('transport',raw),('transport_consensus',consensus)]:
        result[name]=bound_delta((path-components['base'])[None],components['quantiles'][None])[0]/components['std']
    for fraction in [.25,.5,.75]:
        result[f'transport_mix_{fraction}']=(1-fraction)*result['encoder_only']+fraction*result['transport']
    return result


def direct_experts(components):
    """Uniform five-case branch with a foundation-only reference."""
    innovation=components['cases']-components['old_encoded_paths'];mean=innovation.mean(0)
    variance=((innovation-mean)**2).mean(0);confidence=mean**2/(mean**2+variance+1e-12)
    paths=dict(direct_copy=components['cases'].mean(0),direct_transport=components['encoded_path']+mean,
        direct_consensus=components['encoded_path']+confidence*mean,direct_encoder=components['encoded_path'])
    q=components['quantiles'];return {name:bound_delta((path-q[4])[None],q[None])[0]/components['std'] for name,path in paths.items()}
