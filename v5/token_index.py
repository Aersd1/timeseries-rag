"""Disk-backed feasibility indexes: pooled token cosine and token BM25.

Cosine is an explicit block scan, NOT ANN; SQLite BM25 uses inverted postings.
"""
from collections import Counter
import hashlib
import math
import sqlite3
from pathlib import Path
import numpy as np
import torch
from .common import read_json, write_json, fresh_dir, valid_ranges
from .data import Store
from .train import load_checkpoint

TOKEN_DTYPE = np.dtype([('code', '<u2'), ('duration', 'u1'), ('mean', '<f4'), ('std', '<f4')])


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def terms(ids, vocab):
    ids = list(map(int, ids))
    return ids+[vocab+a*vocab+b for a, b in zip(ids[:-1], ids[1:])]


def build(store_path, checkpoint, output, device='cpu'):
    out = fresh_dir(output); store = Store(store_path); model, state = load_checkpoint(checkpoint, device)
    if state['data_id'] != store.meta['data_id']: raise ValueError('Checkpoint belongs to another store')
    c = state['config']; m = c['length']; h = max(c['horizons']); stride = c['index']['stride']
    metadata = []; histogram = np.zeros(c['model']['vocab'], dtype=np.int64)
    with torch.no_grad():
        for sid, s in enumerate(store.series):
            ranges = list(valid_ranges(s, m, h, 0, s['memory_end'], stride))
            total = sum((b-a+step-1)//step for a,b,step in ranges)
            if not total: continue
            if total > c['index']['max_windows_per_series']:
                raise ValueError(f'Series {sid}: {total} library windows exceeds explicit index guard; increase limit or stride in TRAIN config before building')
            folder = out/f'{sid:06d}'; folder.mkdir()
            database = sqlite3.connect(folder/'postings.sqlite')
            database.execute('PRAGMA temp_store=FILE')
            database.execute('CREATE TABLE postings(term INTEGER, doc INTEGER, tf INTEGER, PRIMARY KEY(term,doc)) WITHOUT ROWID')
            writers = {name:(folder/(name+'.bin')).open('wb') for name in ('shape','predictive','tokens','starts')}
            batch_positions = []; count = 0
            def flush():
                nonlocal count
                if not batch_positions: return
                x = np.array([store.window(sid, p, m) for p in batch_positions])
                floor = torch.full((len(x),), max(s['memory_std']*1e-4, 1e-6), device=device)
                result = model(torch.from_numpy(x).to(device), floor)
                ids = result['ids'].cpu().numpy()
                record = np.empty(ids.shape, dtype=TOKEN_DTYPE)
                for name, key in (('code','ids'), ('duration','duration'), ('mean','mu'), ('std','sigma')):
                    record[name] = result[key].cpu().numpy()
                record.tofile(writers['tokens'])
                for name in ('shape','predictive'): result[name].float().cpu().numpy().astype('<f4').tofile(writers[name])
                np.asarray(batch_positions, dtype='<i8').tofile(writers['starts'])
                entries = [(term, count+j, tf) for j, row in enumerate(ids)
                           for term, tf in Counter(terms(row, c['model']['vocab'])).items()]
                database.executemany('INSERT INTO postings VALUES (?,?,?)', entries)
                histogram[:] += np.bincount(ids.reshape(-1), minlength=len(histogram))
                count += len(x); batch_positions.clear(); database.commit()
            try:
                for a,b,step in ranges:
                    for p in range(a,b,step):
                        batch_positions.append(p)
                        if len(batch_positions) == c['index']['batch_size']: flush()
                flush()
                database.execute('CREATE TABLE terms(term INTEGER PRIMARY KEY, idf REAL)')
                reader = database.execute('SELECT term,COUNT(*) FROM postings GROUP BY term')
                while True:
                    rows = reader.fetchmany(4096)
                    if not rows: break
                    database.executemany('INSERT INTO terms VALUES (?,?)',
                                         [(term, math.log1p((count-df+.5)/(df+.5))) for term,df in rows])
                database.commit()
            finally:
                for f in writers.values(): f.close()
                database.close()
            metadata.append(dict(sid=sid, folder=folder.name, windows=count))
            print(f'token index series={sid} windows={count}', flush=True)
    total_windows = sum(s['windows'] for s in metadata)
    j = math.ceil(m/c['model']['primitive']); active = histogram[histogram > 0]
    prob = active/max(1, active.sum()); entropy = float(-np.sum(prob*np.log2(prob)))
    manifest = dict(complete=True, data_id=store.meta['data_id'], checkpoint_sha256=file_hash(checkpoint),
        series=metadata, config=c, windows=total_windows, tokens_per_window=j,
        compression=dict(points_per_token=m/j, token_record_bytes=TOKEN_DTYPE.itemsize,
            raw_window_bytes=m*4, token_window_bytes=j*TOKEN_DTYPE.itemsize,
            raw_to_token_payload_ratio=(m*4)/(j*TOKEN_DTYPE.itemsize),
            index_total_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
            token_entropy_bits=entropy, codebook_perplexity=2**entropy, used_codes=int(len(active)),
            codebook_size=len(histogram), note='Fixed-width records actually stored. Entropy is theoretical; no entropy coder implemented. Overlapping windows duplicate tokens. Index bytes include embeddings and postings.'),
        scope='same-series sampled memory library; stride-1 teacher may contain starts absent from this library')
    write_json(out/'manifest.json', manifest); store.close(); return manifest


class TokenLibrary:
    def __init__(self, path):
        self.path = Path(path); self.meta = read_json(self.path/'manifest.json')
        if not self.meta['complete']: raise ValueError('Incomplete token index')
        self.series = {s['sid']:s for s in self.meta['series']}
        self.current = None

    def select(self, sid):
        if sid == self.current: return
        if self.current is not None: self.database.close()
        s = self.series[sid]; folder = self.path/s['folder']; self.n = s['windows']
        d = self.meta['config']['model']['dim']
        self.starts = np.memmap(folder/'starts.bin', mode='r', dtype='<i8')
        self.shape = np.memmap(folder/'shape.bin', mode='r', dtype='<f4', shape=(self.n,d))
        self.predictive = np.memmap(folder/'predictive.bin', mode='r', dtype='<f4', shape=(self.n,d))
        self.database = sqlite3.connect(f'{(folder/"postings.sqlite").resolve().as_uri()}?mode=ro', uri=True)
        self.database.execute('PRAGMA temp_store=FILE'); self.current = sid

    def cosine(self, vector, k, block=8192):
        ids = np.empty(0, dtype=np.int64); scores = np.empty(0, dtype=np.float32)
        for a in range(0, self.n, block):
            sim = self.shape[a:a+block] @ vector
            take = np.argsort(-sim, kind='stable')[:k]
            all_scores = np.r_[scores, sim[take]]; all_ids = np.r_[ids, a+take]
            order = np.lexsort((all_ids, -all_scores))[:k]
            ids, scores = all_ids[order], all_scores[order]
        return ids

    def bm25(self, codes, k):
        token_terms = sorted(set(terms(codes, self.meta['config']['model']['vocab'])))
        marks = ','.join('?' for _ in token_terms)
        sql = ('SELECT p.doc, SUM(t.idf*p.tf*2.2/(p.tf+1.2)) AS score FROM postings p '
               f'JOIN terms t ON p.term=t.term WHERE p.term IN ({marks}) '
               'GROUP BY p.doc ORDER BY score DESC,p.doc LIMIT ?')
        return np.array([r[0] for r in self.database.execute(sql, [*token_terms, int(k)])], dtype=np.int64)

    def close(self):
        if self.current is not None:
            self.database.close()
            for name in ('starts','shape','predictive'): getattr(self,name)._mmap.close()
            self.current = None
