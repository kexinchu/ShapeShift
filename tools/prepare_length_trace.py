"""Convert a local length CSV into explicit replay input; never invent real session timing."""
import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from shapeshift.replay import load_trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--gen-length',type=int,default=128)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--preserve-arrivals',action='store_true')
    group.add_argument('--constant-gap-ms',type=float)
    args=p.parse_args()
    import math
    if args.gen_length<1 or (args.constant_gap_ms is not None and
                            (not math.isfinite(args.constant_gap_ms) or args.constant_gap_ms<0)):
        p.error('invalid length/gap')
    with args.input.open(newline='') as f:
        rows=[r for r in csv.DictReader(f) if r.get('source',r.get('task'))==args.source]
    if not rows:
        p.error('no matching rows')
    out=[]
    for i,r in enumerate(rows):
        g=int(r.get('gen_len') or args.gen_length)
        if g!=args.gen_length:
            p.error('generation length mismatch')
        length=int(r['prompt_len'])
        if r.get('total_len') and int(r['total_len'])!=length+g:
            p.error('inconsistent total_len')
        out.append(dict(source=args.source,request_uid=r.get('request_uid') or str(i),
            prompt_len=length,gen_len=g,total_len=length+g,
            arrival_ms=r['arrival_ms'] if args.preserve_arrivals else i*args.constant_gap_ms,
            arrival_origin='source' if args.preserve_arrivals else 'synthetic_constant'))
    # Validate before producing the destination; never overwrite user data.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmpfile=Path(tmp)/'trace.csv'
        with tmpfile.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
        load_trace(tmpfile,gen_length=args.gen_length)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('xb') as f:f.write(tmpfile.read_bytes())
    print(f'wrote {len(out)} ordered requests; timing origin: {out[0]["arrival_origin"]}')


if __name__=='__main__':main()
