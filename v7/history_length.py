"""Selection and diagnostics for a fixed frozen-forecaster context sweep."""
import numpy as np

LENGTHS=(96,512,2016,4096)


def observed_context(values,history_start,length):
    end=int(history_start)+96;begin=end-int(length)
    if begin<0:raise ValueError('Insufficient observed history')
    x=np.array(values[begin:end],dtype=np.float32,copy=True)
    if x.shape!=(length,) or not np.isfinite(x).all():raise ValueError('Incomplete or gapped context')
    return x


def common_eligible(series,start,length=4096):
    left=int(start)+96-length;right=int(start)+192
    return left>=0 and any(a<=left and right<=b for a,b in series['runs'])


def metrics(y,p,sd):
    error=np.asarray(p,dtype=float)-np.asarray(y,dtype=float)
    return dict(mse=float(np.mean(error**2)),mae=float(np.mean(abs(error))),
                train_nmse=float(np.mean((error/np.asarray(sd)[:,None])**2)))


def block_metrics(data,predictions):
    chunks=np.array_split(np.unique(data['start']),4);result=[]
    for i,starts in enumerate(chunks):
        ids=np.flatnonzero(np.isin(data['start'],starts))
        if not len(ids):continue
        result.append(dict(block=i+1,first_start=int(starts.min()),last_start=int(starts.max()),
                           origins=len(starts),queries=len(ids),
                           metrics={str(k):metrics(data['raw_y'][ids],p[ids],data['train_std'][ids]) for k,p in predictions.items()}))
    return result


def select_lengths(validation_metrics,validation_blocks):
    # No TEST argument, rejection search, or dataset-wise TEST winner is permitted.
    pooled=min(LENGTHS,key=lambda k:(validation_metrics[str(k)]['mse'],k))
    worst={str(k):max(b['metrics'][str(k)]['mse']/max(b['metrics']['96']['mse'],1e-20)
                      for b in validation_blocks) for k in LENGTHS}
    robust=min(LENGTHS,key=lambda k:(worst[str(k)],k))
    return dict(primary_length=pooled,robust_secondary_length=robust,
                primary_rule='Minimum pooled VAL raw MSE; ties choose shorter history',
                robust_rule='Minimum worst of four VAL block MSE ratios versus96; diagnostic secondary only',
                validation_metrics=validation_metrics,validation_worst_block_ratios=worst)


def paired_interval(data,base,pred,period):
    b=np.mean((base.astype(float)-data['raw_y'])**2,1)
    p=np.mean((pred.astype(float)-data['raw_y'])**2,1)
    week=(data['start']+96)//max(1,int(round(7*(period or 1))))
    losses=np.array([[b[week==w].sum(),p[week==w].sum()] for w in np.unique(week)])
    rng=np.random.default_rng(20261016)
    sampled=losses[rng.integers(0,len(losses),size=(4000,len(losses)))].sum(1)
    gain=100*(1-sampled[:,1]/np.maximum(sampled[:,0],1e-20))
    return dict(gain_percent=float(100*(1-p.mean()/max(b.mean(),1e-20))),
                exploratory_interval95=np.quantile(gain,[.025,.975]).tolist(),weekly_clusters=len(losses))


def plateau_diagnostic(comparison,margin=1.):
    lo,hi=comparison['exploratory_interval95'];point=comparison['gain_percent']
    if lo>=-margin and hi<=margin:label='within_predeclared_one_percent_band'
    elif lo>0 and point>=margin:label='4096_still_improves_over2016'
    elif hi<0:label='4096_worse_than2016'
    else:label='insufficient_evidence_of_plateau'
    return dict(classification=label,relative_mse_margin_percent=margin,
                caveat='Only this model and these lengths; nominal exploratory interval, not proof that information is absent')
