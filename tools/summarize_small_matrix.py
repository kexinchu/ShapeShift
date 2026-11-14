import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shapeshift.matrix_summary import summarize

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output_root',type=Path)
    args=p.parse_args()
    report=summarize(args.output_root)
    print(json.dumps(report,indent=2,allow_nan=False))
    raise SystemExit(0 if report['valid_for_comparison'] else 2)
