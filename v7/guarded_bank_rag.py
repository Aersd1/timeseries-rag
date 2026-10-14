"""Select one five-case branch against a fixed incumbent, then confirm once."""
import numpy as np

FAMILIES=['future_state_rag','mix_0.5','transport','transport_consensus']


def choose_bank(validation,original,incumbent,encoder_policy,direct=False,no_retrieval=False):
    sid=original['sid'];start=original['start'];policies={}
    for variable in np.unique(sid):
        key=str(int(variable));rows=np.flatnonzero(sid==variable);rows=rows[np.argsort(start[rows])];parts=np.array_split(np.arange(len(rows)),3)
        p=incumbent[key];baseline=(original['routed'][rows]-original['base'][rows])/original['std'][rows,None]
        target=(original['raw_y'][rows]-original['base'][rows])/original['std'][rows,None]
        default=dict(branch='incumbent',family='routed',beta=0.,reason='no confirmed incremental advantage')
        if min(map(len,parts))<3:
            policies[key]=default;continue
        errors=target-baseline;options=[]
        for branch,table in validation.items():
            allowed=['direct_encoder'] if no_retrieval else FAMILIES+(['direct_copy','direct_transport','direct_consensus'] if direct else [])
            for family in allowed:
                independent=family.startswith('direct_')
                # Only a rejected BASE invalidates adapter-dependent paths. A prior
                # retrieval branch's rejection must not suppress a new candidate
                # combination selected before its single confirmation block.
                if not independent and np.any(original['disabled'][rows]):continue
                delta=table[f'expert_{family}'][rows]
                for beta in [.1,.25,.5]:
                    offset=(original['moirai_direct'][rows]-original['base'][rows])/original['std'][rows,None] if independent else 0.
                    error=target-offset-beta*delta;stable=True;gains=[]
                    for ids in parts[:2]:
                        gain=np.mean(errors[ids]**2,axis=1)-np.mean(error[ids]**2,axis=1)
                        if gain.mean()<=.01*np.mean(errors[ids]**2) or np.mean(gain>0)<.75 or np.mean(abs(error[ids]))>=np.mean(abs(errors[ids])):
                            stable=False;break
                        gains.extend(gain.tolist())
                    if stable:options.append((-float(np.mean(gains)),branch,family,beta))
        if not options:policies[key]=default;continue
        _,branch,family,beta=min(options);held=parts[2]
        offset=(original['moirai_direct'][rows]-original['base'][rows])/original['std'][rows,None] if family.startswith('direct_') else 0.
        error=target-offset-beta*validation[branch][f'expert_{family}'][rows]
        gain=np.mean(errors[held]**2,axis=1)-np.mean(error[held]**2,axis=1)
        confirmed=bool(gain.mean()>0 and np.mean(gain>0)>=.75 and np.mean(abs(error[held]))<np.mean(abs(errors[held])))
        policies[key]=dict(branch=branch if confirmed else 'incumbent',family=family if confirmed else 'routed',beta=beta if confirmed else 0.,
            proposed_branch=branch,proposed_family=family,proposed_beta=beta,final_gain=float(gain.mean()),final_win_fraction=float(np.mean(gain>0)),
            reason='single five-case branch confirmed' if confirmed else 'selected branch rejected; preserve incumbent')
    # Same encoder/no-retrieval: fixed original predictor, no new forecasting fit.
    # Branches share exactly this encoder; choosing a retrieval branch adds no model capacity.
    return policies


def apply_bank(tables,original,policy):
    output=original['routed'].copy();selected=original['selected_starts'].copy()
    for sid in np.unique(original['sid']):
        p=policy[str(int(sid))];ids=original['sid']==sid
        if p['branch']=='incumbent':continue
        current=tables[p['branch']]
        base=original['moirai_direct'] if p['family'].startswith('direct_') else original['final_base']
        output[ids]=base[ids]+p['beta']*current[f"expert_{p['family']}"][ids]*original['std'][ids,None]
        selected[ids]=current['selected_starts'][ids]
    return output,selected
