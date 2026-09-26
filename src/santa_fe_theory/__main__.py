import argparse
from .io import export_run
from .branch import export_branches

def main():
    parser=argparse.ArgumentParser(description='Reduced Santa Fe hybrid Langevin theory')
    subs=parser.add_subparsers(dest='command',required=True)
    run=subs.add_parser('run');run.add_argument('--config',required=True);run.add_argument('--initials',required=True);run.add_argument('--out',required=True)
    branch=subs.add_parser('branch');branch.add_argument('--config',required=True);branch.add_argument('--snapshots',required=True);branch.add_argument('--pairs',type=int,default=256);branch.add_argument('--out',required=True)
    compare=subs.add_parser('compare');compare.add_argument('--theory',required=True);compare.add_argument('--simulation',required=True);compare.add_argument('--simulation-manifest',required=True);compare.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.command=='run':out=export_run(args.config,args.initials,args.out)
    elif args.command=='branch':out=export_branches(args.config,args.snapshots,args.out,args.pairs)
    else:
        from .compare import compare_exports
        out=compare_exports(args.theory,args.simulation,args.simulation_manifest,args.out)
    print(out)
if __name__=='__main__':main()
