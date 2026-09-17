"""History-neighbor batches: learn to separate plausible but different futures.

Only observed histories form the neighbor tree. No all-pairs distances are saved.
"""
import math
import numpy as np
from scipy.spatial import cKDTree
from .data import scale_floor


class HistoryBatches:
    def __init__(self,dataset,batch_size,seed,neighbor_fraction=.5):
        self.dataset,self.batch_size,self.seed,self.epoch=dataset,batch_size,seed,0
        if batch_size<3 or not 0<neighbor_fraction<1: raise ValueError('Invalid neighbor batch settings')
        c=dataset.c; self.span=c['length']+max(c['horizons'])
        self.neighbors=max(1,int(batch_size*neighbor_fraction))
        features=[]
        for sid,start in dataset.refs:
            x=dataset.store.window(sid,start,c['length']).astype(float)
            z=(x-x.mean())/max(float(x.std()),scale_floor(dataset.store.series[sid],c))
            pooled=np.array([part.mean() for part in np.array_split(z,min(c['length'],c['model']['history_bins']))])
            features.append(pooled/max(float(np.linalg.norm(pooled)),1e-8))
        self.features=np.asarray(features); self.tree=cKDTree(self.features)

    def __len__(self): return math.ceil(len(self.dataset)/self.batch_size)

    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch); n=len(self.dataset)
        anchors=rng.permutation(n)[:len(self)]
        for anchor in anchors:
            sid,start=self.dataset.refs[int(anchor)]
            _,near=self.tree.query(self.features[anchor],k=min(n,self.batch_size*8))
            batch=[int(anchor)]
            for j in np.atleast_1d(near):
                j=int(j); other_sid,other_start=self.dataset.refs[j]
                if j!=anchor and (other_sid!=sid or abs(other_start-start)>=self.span):
                    batch.append(j)
                    if len(batch)>self.neighbors: break
            # Random remainder keeps broad negatives and prevents local-only batches.
            used=set(batch)
            for j in rng.permutation(n):
                if len(batch)>=min(n,self.batch_size): break
                if int(j) not in used: batch.append(int(j)); used.add(int(j))
            yield batch
