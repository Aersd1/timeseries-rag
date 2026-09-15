"""Storage estimates only. Deliberately does NOT extrapolate query latency."""
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',default='results/validation_paa64')
    ap.add_argument('--points',type=int,default=10000000000);ap.add_argument('--workers',type=int,default=16)
    args=ap.parse_args();path=Path(args.index);m=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    if args.points<1 or args.workers<1:raise ValueError('Positive points and worker counts required')
    components=dict(raw=0,row_maps=0,codes=0,positions=0,trees=0,metadata=0)
    for p in path.rglob('*'):
        if not p.is_file() or 'staging' in p.parts or p.name.startswith('checkpoint'):continue
        name=p.name
        bucket='raw' if name=='raw.bin' else 'row_maps' if name=='rows.bin' else 'codes' if name=='codes.npy' else 'positions' if name=='records.npy' else 'trees' if p.suffix=='.npy' else 'metadata'
        components[bucket]+=p.stat().st_size
    points=m.get('unique_raw_points',m['stored_points']);factor=args.points/points
    projected={k:v*factor for k,v in components.items()};total=sum(projected.values())
    model=json.loads((path/'model/model.json').read_text());dims=model['pairs']*2+model['paa']
    scratch=(dims+25)*m['records']*factor
    result=dict(measured_unique_points=points,measured_windows=m['windows'],measured_records=m['records'],
        measured_components_bytes=components,target_original_points=args.points,workers=args.workers,
        projected_components_bytes=projected,projected_steady_bytes=total,
        projected_build_peak_bytes=total+scratch,
        projected_local_index_bytes_per_worker=(total-projected['raw']-projected['row_maps'])/args.workers,
        raw_data_storage_options=dict(shared_raw_store_bytes=projected['raw']+projected['row_maps'],
            raw_replica_bytes_per_worker=projected['raw']+projected['row_maps']),
        uncompressed_feature_and_position_upper_bytes=args.points*(dims+25),
        assumptions=['Extrapolates this corpus run-compression and missing/segment distribution.',
            'Not an actual 10B-point build or latency benchmark.',
            'Current workers share one completed index/raw store; remote storage performance is untested.',
            'Build retains staging until atomic publication; reserve scratch space.',
            'No latency estimate is made from a million-point benchmark.'])
    out=path.parent/'capacity_10B.json';out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
