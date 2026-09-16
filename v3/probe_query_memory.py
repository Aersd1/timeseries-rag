"""Measure incremental traced allocations after index startup, not full RSS."""
import json,tracemalloc
from pathlib import Path
from diverse import DiverseIndex

def main():
    engine=DiverseIndex('../results/cascade_index')
    q=json.loads(Path('../results/query.json').read_text(encoding='utf-8'))['values']
    results=[]
    try:
        for count in (128,512,2048,4384):
            tracemalloc.start()
            result=engine.search_bounded(q,initial_candidates=count,max_candidates=8192)
            current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            results.append(dict(initial=count,current_traced_bytes=current,peak_traced_bytes=peak,
                retained=result['max_retained_candidates'],candidate_payload_cap_bytes=result['candidate_payload_bound_bytes'],
                block_distances=result['peak_block_distance_count'],complete=result['complete']))
            del result
        out=Path('results/redundancy/memory_probe.json');out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(dict(measurement='tracemalloc incremental peak after index ready; not process RSS; native untracked allocations may be excluded; no audit distance arrays',results=results),indent=2),encoding='utf-8')
        print(json.dumps(results,indent=2))
    finally:engine.close()

if __name__=='__main__':main()
