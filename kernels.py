import ctypes,os
from pathlib import Path
import numpy as np
LIB=ctypes.CDLL(str(Path(__file__).with_name('native.dll' if os.name=='nt' else 'native.so')))
F=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS')
I=np.ctypeslib.ndpointer(np.int64,flags='C_CONTIGUOUS')
U=np.ctypeslib.ndpointer(np.uint8,flags='C_CONTIGUOUS')
L=ctypes.c_int64; C=ctypes.c_int; D=ctypes.c_double
LIB.encode.argtypes=[F,L,C,I,C,C,F,U,U,ctypes.c_void_p]
LIB.exact_ed.argtypes=[F,I,L,C,F,I,D,F]
LIB.exact_dtw.argtypes=[F,I,L,C,F,F,F,C,D,F,I]
LIB.bounds.argtypes=[U,U,L,C,C,F,F,F,F,C,F]
LIB.search_tree.argtypes=[F,I,U,U,I,I,U,U,I,F,F,F,F,F,I,F,F,F,I,I,I,I,D,I,F]
LIB.search_tree.restype=C
LIB.window_stats.argtypes=[F,L,C,F,F]
LIB.window_stats.restype=None
LIB.filter_codes.argtypes=[U,L,C,C,F,D,I]
LIB.filter_codes.restype=L
LIB.local_stats.argtypes=[F,L,C,F,F]
LIB.local_stats.restype=None
LIB.exact_ed_cached.argtypes=[F,I,L,C,F,I,F,F,D,F]
LIB.exact_ed_cached.restype=None
for name in ('encode','exact_ed','exact_dtw','bounds'):
    getattr(LIB,name).restype=None

def encode(x,model,debug=False):
    x=np.ascontiguousarray(x,dtype=float); n=len(x)-model.length+1
    codes=np.empty((n,model.dim),dtype=np.uint8); flat=np.empty(n,dtype=np.uint8)
    values=np.empty((n,model.dim)) if debug else None
    LIB.encode(x,len(x),model.length,model.freq,len(model.freq),model.paa,model.bins,
        codes,flat,None if values is None else values.ctypes.data)
    return codes,flat,values

def bounds(lo,hi,model,ql,qu,metric='ed'):
    lo=np.ascontiguousarray(lo); hi=np.ascontiguousarray(hi); out=np.empty(len(lo))
    LIB.bounds(lo,hi,len(lo),model.dim,2*len(model.freq),model.left,model.right,
        np.ascontiguousarray(ql),np.ascontiguousarray(qu),metric=='dtw',out)
    return out

def exact(x,starts,zq,metric='ed',radius=8,cutoff=np.inf,envelope=None):
    x=np.ascontiguousarray(x); starts=np.ascontiguousarray(starts,dtype=np.int64)
    zq=np.ascontiguousarray(zq); out=np.empty(len(starts)); stats=np.zeros(2,dtype=np.int64)
    if metric=='ed':
        order=np.ascontiguousarray(np.argsort(-abs(zq)),dtype=np.int64)
        LIB.exact_ed(x,starts,len(starts),len(zq),zq,order,cutoff,out)
        stats[:]=len(starts)
    else:
        lower,upper=envelope
        LIB.exact_dtw(x,starts,len(starts),len(zq),zq,np.ascontiguousarray(lower),
            np.ascontiguousarray(upper),radius,cutoff,out,stats)
    return out,stats

def search_tree(raw,offsets,tree,model,ql,qu,zq,metric,radius,best,sids,positions,count,constant_count,budget_ms,envelope):
    config=np.array([len(tree['lo']),len(tree['records']),model.dim,model.pairs*2,model.length,metric=='dtw',radius,len(best)],dtype=np.int64)
    order=np.ascontiguousarray(np.argsort(-abs(zq)),dtype=np.int64)
    envlo,envhi=(zq,zq) if envelope is None else envelope
    stats=np.zeros(6,dtype=np.int64);frontier=np.empty(1)
    status=LIB.search_tree(raw,offsets,tree['lo'],tree['hi'],tree['children'],tree['spans'],tree['codes'],
        tree['records'].view(np.uint8),config,model.left,model.right,np.ascontiguousarray(ql),np.ascontiguousarray(qu),
        np.ascontiguousarray(zq),order,np.ascontiguousarray(envlo),np.ascontiguousarray(envhi),best,sids,positions,count,
        constant_count,-1. if budget_ms is None else max(0.,budget_ms),stats,frontier)
    if status<0:raise MemoryError('Native tree heap allocation failed')
    return bool(status),stats,float(frontier[0])

def window_stats(x,m):
    x=np.ascontiguousarray(x);mean=np.empty(len(x)-m+1);sd=np.empty_like(mean)
    LIB.window_stats(x,len(x),m,mean,sd);return mean,sd

def filter_codes(codes,lookup,nsfa,cutoff):
    codes=np.ascontiguousarray(codes);out=np.empty(len(codes),dtype=np.int64)
    n=LIB.filter_codes(codes,len(codes),codes.shape[1],nsfa,np.ascontiguousarray(lookup),cutoff,out)
    return out[:n]

def local_stats(x,m):
    x=np.ascontiguousarray(x);mean=np.empty(len(x)-m+1);sd=np.empty_like(mean)
    LIB.local_stats(x,len(x),m,mean,sd);return mean,sd

def exact_cached(x,starts,zq,stats,cutoff=np.inf):
    starts=np.ascontiguousarray(starts,dtype=np.int64);out=np.empty(len(starts))
    order=np.ascontiguousarray(np.argsort(-abs(zq)),dtype=np.int64)
    LIB.exact_ed_cached(np.ascontiguousarray(x),starts,len(starts),len(zq),np.ascontiguousarray(zq),order,
        stats[0],stats[1],cutoff,out)
    return out
