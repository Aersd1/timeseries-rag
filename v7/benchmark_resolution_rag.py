"""Train-only selection of future-state resolution, with matched no-RAG controls."""
import argparse
import json
from pathlib import Path
from .benchmark_future_state_rag import experiment as joint
from .benchmark_group_future_state import experiment as grouped


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-root',required=True);p.add_argument('--source',default='v7/runs/all_datasets_20261008')
    p.add_argument('--output',required=True);p.add_argument('--datasets',nargs='+',required=True);a=p.parse_args()
    for name in a.datasets:
        previous=Path(a.checkpoint_root)/name;run=json.loads((previous/'run.json').read_text(encoding='utf-8'))
        source=Path(a.source)/name;out=Path(a.output)/name;base=Path(run['source_checkpoint'])
        if run.get('grouped'):grouped(base,source,out,optimize_resolution=True)
        else:joint(base,Path(run['source_complementary']),source,out,optimize_resolution=True)
