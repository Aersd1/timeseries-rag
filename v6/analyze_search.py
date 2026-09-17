"""Regenerate a portable report/plot from a completed retrieval benchmark."""
import argparse
import json
import shutil
from pathlib import Path
from .benchmark_search import report


def analyze(results, output):
    results,output=Path(results),Path(output)
    value=json.loads((results/'benchmark.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True,exist_ok=True)
    if results.resolve()!=output.resolve(): shutil.copy2(results/'benchmark.json',output/'benchmark.json')
    report(value,output)
    print(output/'REPORT.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); analyze(a.results,a.output)
