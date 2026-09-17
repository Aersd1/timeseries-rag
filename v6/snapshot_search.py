"""Save a trusted Git revision's search sources for a local paired benchmark."""
import argparse
import subprocess
from pathlib import Path


def snapshot(ref, output):
    root=Path(__file__).resolve().parents[1]
    commit=subprocess.check_output(['git','rev-parse','--verify',ref+'^{commit}'],cwd=root,text=True).strip()
    sources={name:subprocess.check_output(['git','show',f'{commit}:v6/{name}'],cwd=root)
             for name in ('index.py','inference.py')}
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    for name,value in sources.items(): (output/('baseline_'+name)).write_bytes(value)
    (output/'baseline_commit.txt').write_text(commit+'\n',encoding='utf-8')
    print(commit)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ref',required=True,help='Trusted local commit whose code will run in the benchmark')
    p.add_argument('--output',required=True)
    a=p.parse_args(); snapshot(a.ref,a.output)
