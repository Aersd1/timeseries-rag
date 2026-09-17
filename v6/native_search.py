"""Optional C heap; NumPy fallback remains usable without a compiler."""
import ctypes
import os
from pathlib import Path
import numpy as np

LIB = None
LOAD_ERROR = None
try:
    LIB = ctypes.CDLL(str(Path(__file__).with_name('native.dll' if os.name == 'nt' else 'native.so')))
    # Wrapper below validates and owns contiguous arrays. Retain buffer addresses
    # to avoid six repeated ctypes ndarray inspections on every leaf.
    F = I = ctypes.c_void_p
    L = ctypes.c_int64
    LIB.candidate_add.argtypes = [F,I,L,L,L,F,I,I,I]
    LIB.candidate_add.restype = None
except (OSError, AttributeError) as exc:
    LIB = None
    LOAD_ERROR = str(exc)


class NativeCandidates:
    def __init__(self, capacity):
        if not isinstance(capacity, (int,np.integer)) or capacity < 1:
            raise ValueError('Positive integer candidate capacity required')
        if LIB is None:
            raise RuntimeError('Build optional V6 native heap with python -m v6.build_native: '+str(LOAD_ERROR))
        self.capacity = capacity
        self.distances = np.empty(capacity, dtype=np.float64)
        self.sids = np.empty(capacity, dtype=np.int64)
        self.starts = np.empty(capacity, dtype=np.int64)
        self.count = np.zeros(1, dtype=np.int64)
        self._buffers=(self.distances.ctypes.data,self.sids.ctypes.data,
                       self.starts.ctypes.data,self.count.ctypes.data)

    def __len__(self):
        return int(self.count[0])

    @property
    def threshold(self):
        return float(self.distances[0]) if len(self) == self.capacity else float('inf')

    def add(self, distances, sid, starts):
        ds = np.ascontiguousarray(distances, dtype=np.float64)
        ps = np.ascontiguousarray(starts, dtype=np.int64)
        if ds.ndim != 1 or ds.shape != ps.shape:
            raise ValueError('Candidate distances/starts must be matching vectors')
        LIB.candidate_add(ds.ctypes.data, ps.ctypes.data, len(ds), int(sid), self.capacity,*self._buffers)

    def hits(self):
        n = len(self)
        order = np.lexsort((self.starts[:n], self.sids[:n], self.distances[:n]))
        return [dict(distance=d,sid=s,start=p) for d,s,p in zip(
            self.distances[order].tolist(),self.sids[order].tolist(),self.starts[order].tolist())]
