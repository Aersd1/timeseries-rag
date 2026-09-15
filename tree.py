"""Bulk-built region trees; node boxes enclose every record, not just a centroid."""
import numpy as np

def build_tree(lo,hi=None,leaf_size=256):
    hi=lo if hi is None else hi
    order=np.arange(len(lo),dtype=np.int64); lows=[]; highs=[]; children=[]; spans=[]
    def visit(a,b):
        node=len(lows); ids=order[a:b]
        low=lo[ids].min(axis=0); high=hi[ids].max(axis=0)
        lows.append(low);highs.append(high);children.append([-1,-1]);spans.append([a,b])
        if b-a<=leaf_size or np.array_equal(low,high):
            return node
        spread=high.astype(np.int16)-low.astype(np.int16)
        axis=int(np.argmax(spread))
        values=lo[ids,axis].astype(np.int16)+hi[ids,axis]
        mid=(b-a)//2; part=np.argpartition(values,mid)
        order[a:b]=ids[part]
        children[node]=[visit(a,a+mid),visit(a+mid,b)]
        return node
    if len(lo):visit(0,len(lo))
    return dict(order=order,lo=np.asarray(lows,dtype=np.uint8),hi=np.asarray(highs,dtype=np.uint8),
        children=np.asarray(children,dtype=np.int64),spans=np.asarray(spans,dtype=np.int64))

def save_tree(path,tree):
    path.mkdir(parents=True,exist_ok=True)
    for key,value in tree.items():np.save(path/f'{key}.npy',value)

def load_tree(path):
    return {key:np.load(path/f'{key}.npy',mmap_mode='r') for key in ('order','lo','hi','children','spans')}
