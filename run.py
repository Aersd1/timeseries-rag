import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,time
from pathlib import Path
from model import LearnedShapeModel
from index import IndexBuilder,SearchIndex
from data import inventory,csv_runs,validation_data,training_windows

def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('build-validation');demo.add_argument('--output',default='results/validation_index')
    demo.add_argument('--project',default='..');demo.add_argument('--gpu',default=r'D:\workload\gpu')
    demo.add_argument('--paa',type=int,default=16);demo.add_argument('--pairs',type=int,default=16)
    build=sub.add_parser('build-csv');build.add_argument('--output',required=True);build.add_argument('--model',required=True)
    build.add_argument('--project',default='..');build.add_argument('--gpu',default=r'D:\workload\gpu')
    build.add_argument('--limit-files',type=int,default=0);build.add_argument('--max-points',type=int,default=0)
    build.add_argument('--resume',action='store_true')
    query=sub.add_parser('query');query.add_argument('--index',required=True);query.add_argument('--query-json',required=True)
    query.add_argument('--metric',choices=('ed','dtw','robust'),default='ed');query.add_argument('--radius',type=int,default=8)
    query.add_argument('--budget-ms',type=float);query.add_argument('--k',type=int,default=10);query.add_argument('--output')
    args=ap.parse_args()
    if args.command=='build-validation':
        parts,scope=validation_data(args.project,args.gpu)
        model=LearnedShapeModel(paa=args.paa,pairs=args.pairs).fit(training_windows(parts))
        builder=IndexBuilder(args.output,model)
        for i,(x,meta,rows) in enumerate(parts):
            builder.add(x,meta,rows)
            if (i+1)%10==0:print(f'{i+1}/{len(parts)} runs; {builder.windows:,} positions',flush=True)
        manifest=builder.finish(scope);print(json.dumps({k:v for k,v in manifest.items() if k not in ('series','shards','scope')},indent=2));return
    if args.command=='build-csv':
        files=inventory(args.project,args.gpu);selected=files[:args.limit_files] if args.limit_files else files
        builder=IndexBuilder(args.output,LearnedShapeModel.load(args.model),resume=args.resume)
        processed=list(builder.processed_files);known={f['path']:f for f in processed};limited=False
        for file in selected:
            if file['path'] in known:
                old=known[file['path']]
                if old['bytes']!=file['bytes'] or old['mtime_ns']!=file['mtime_ns']:raise ValueError('An indexed source changed')
                continue
            complete_file=True
            for x,meta,rows in csv_runs(file):
                fresh=builder.new_points(meta,len(x))
                if args.max_points and builder.unique_points+fresh>args.max_points:
                    keep=len(x)-(builder.unique_points+fresh-args.max_points)
                    if keep>=244:builder.add(x[:keep],meta,rows[:keep])
                    limited=True;complete_file=False;break
                builder.add(x,meta,rows)
            processed.append(dict(file,complete_file=complete_file))
            builder.processed_files=processed;builder.checkpoint()
            print(f'{len(processed)}/{len(files)} files; {builder.unique_points:,} unique points; {builder.windows:,} positions',flush=True)
            if limited:break
        manifest=builder.finish(dict(files=processed,total_files=len(files),full_requested_corpus=not limited and len(processed)==len(files)))
        print(f"Completed this index: {manifest['windows']:,} positions");return
    idx=SearchIndex(args.index);data=json.loads(Path(args.query_json).read_text(encoding='utf-8'))
    q=data if isinstance(data,list) else data['values']
    options=dict(k=args.k,radius=args.radius,time_budget_ms=args.budget_ms,include_values=True)
    result=idx.robust_search(q,**options) if args.metric=='robust' else idx.search(q,metric=args.metric,**options)
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:Path(args.output).write_text(text,encoding='utf-8')
    print(text);idx.close()

if __name__=='__main__':main()
