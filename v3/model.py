import hashlib,json
from pathlib import Path
import numpy as np
from scipy.ndimage import minimum_filter1d,maximum_filter1d

def znorm(x):
    x=np.asarray(x,dtype=float)
    y=x-x[..., :1]; y=y-y.mean(axis=-1,keepdims=True)
    sd=np.sqrt(np.mean(y*y,axis=-1,keepdims=True))
    return np.where(sd>1e-10,y/np.maximum(sd,1e-10),0.)

class LearnedShapeModel:
    def __init__(self,length=244,pairs=16,paa=16):
        if not 1<=pairs<length//2 or not 1<=paa<=length:
            raise ValueError('Invalid model dimensions')
        self.length,self.pairs,self.paa=length,pairs,paa
        self.edges=np.arange(paa+1)*length//paa
        self.freq=np.arange(1,pairs+1,dtype=np.int64)
        self.dim=pairs*2+paa

    def transform(self,x):
        z=znorm(x); fft=np.fft.rfft(z,axis=-1)[...,self.freq]*np.sqrt(2/self.length)
        sfa=np.stack([fft.real,fft.imag],axis=-1).reshape(*z.shape[:-1],self.pairs*2)
        paa=np.stack([z[...,a:b].sum(axis=-1)/np.sqrt(b-a) for a,b in zip(self.edges[:-1],self.edges[1:])],axis=-1)
        return np.concatenate([sfa,paa],axis=-1)

    def fit(self,windows,seed=20260915):
        w=np.asarray(windows,dtype=float)
        if w.ndim!=2 or w.shape[1]!=self.length or len(w)<10 or not np.isfinite(w).all():
            raise ValueError('Need at least 10 finite training windows')
        f=np.fft.rfft(znorm(w),axis=-1)*np.sqrt(2/self.length)
        variance=np.var(f.real,axis=0)+np.var(f.imag,axis=0)
        eligible=np.arange(1,self.length//2,dtype=np.int64)
        self.freq=np.ascontiguousarray(eligible[np.argsort(-variance[eligible],kind='stable')[:self.pairs]])
        values=self.transform(w)
        self.bins=np.ascontiguousarray(np.quantile(values,np.arange(1,256)/256,axis=0).T)
        self.left=np.ascontiguousarray(np.c_[np.full(self.dim,-np.inf),self.bins])
        self.right=np.ascontiguousarray(np.c_[self.bins,np.full(self.dim,np.inf)])
        self.training=dict(samples=len(w),seed=seed,selected_frequencies=self.freq.tolist(),
            training_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
            method='variance-selected orthogonal Fourier pairs + learned quantile bins; fixed PAA safety channel')
        return self

    def query(self,q,metric='ed',radius=8):
        q=np.asarray(q,dtype=float)
        if q.shape!=(self.length,) or not np.isfinite(q).all():
            raise ValueError(f'Expected {self.length} finite query values')
        if metric not in ('ed','dtw') or not 0<=radius<self.length:
            raise ValueError('Invalid metric or warping radius')
        z=znorm(q); feature=self.transform(q)
        if metric=='ed':
            return z,feature,feature,None
        lower=minimum_filter1d(z,size=2*radius+1,mode='nearest')
        upper=maximum_filter1d(z,size=2*radius+1,mode='nearest')
        ql=feature.copy(); qu=feature.copy()
        for j,(a,b) in enumerate(zip(self.edges[:-1],self.edges[1:])):
            ql[2*self.pairs+j]=lower[a:b].sum()/np.sqrt(b-a)
            qu[2*self.pairs+j]=upper[a:b].sum()/np.sqrt(b-a)
        return z,ql,qu,(lower,upper)

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        np.savez(path/'model.npz',freq=self.freq,bins=self.bins)
        (path/'model.json').write_text(json.dumps(dict(length=self.length,pairs=self.pairs,paa=self.paa,
            training=self.training),ensure_ascii=False,indent=2),encoding='utf-8')

    @classmethod
    def load(cls,path):
        path=Path(path); info=json.loads((path/'model.json').read_text(encoding='utf-8'))
        obj=cls(info['length'],info['pairs'],info['paa']); values=np.load(path/'model.npz')
        obj.freq=np.ascontiguousarray(values['freq']);obj.bins=np.ascontiguousarray(values['bins'])
        obj.left=np.ascontiguousarray(np.c_[np.full(obj.dim,-np.inf),obj.bins])
        obj.right=np.ascontiguousarray(np.c_[obj.bins,np.full(obj.dim,np.inf)])
        obj.training=info['training'];return obj
