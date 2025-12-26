import os
import time
import subprocess
import numpy as np
from scipy.spatial import KDTree
import cupy as cp

from em3na.utils.misc_utils import pjoin, abspath
from em3na.utils.torch_utils import get_device_names
from em3na.ms.getp import getp

def distance(x, y):
    # x, y - (*, d)
    return np.sqrt(np.sum(np.power((x-y), 2), axis=-1))


def run_getp(map_dir, out_dir, lib_dir, pdb=None, res=6.0, thresh=20, nt=4, filter=0.0, dmerge=1.0, verbose=False, **kwargs):
    # ms using compiled binary
    cmd = "{}/bin/getp --in {} --out {} --thresh {} --res {} --nt {} --filter {} --dmerge {}".format(lib_dir, map_dir, out_dir, thresh, res, nt, filter, dmerge)

    if pdb is not None:
        cmd += " --pdb {}".format(pdb)

    if verbose:
        print(f"# Running command {cmd}")

    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print("# Error at getp")
        print("# Original stdout:", flush=True)
        print(result.stdout)
        print("# Original stderr:", flush=True)
        print(result.stderr)
        exit(1)

    return True



def main(args):
    ts = time.time()

    out_dir = abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)

    # Other params
    res = args.res
    thresh = args.thresh
    nt = args.nt
    filter = args.filter
    dmerge = args.dmerge

    assert 2.0 <= res <= 8.0, "2.0 <= res <= 8.0 but got {:.2f}".format(res)
    assert thresh >= 0.0, "thresh >= 0.0 but got {:.2f}".format(thresh)
    assert nt >= 1, "nt >= 1 but got {}".format(nt)

    getp_args = {
        'res': res,
        'thresh': thresh,
        'nt': nt,
        'filter': filter,
        'dmerge': dmerge,
    }
    print(
        "# Running getp with args res={} thresh={} nt={}".format(
        getp_args['res'],
        getp_args['thresh'],
        getp_args['nt'],
        getp_args['filter'],
        getp_args['dmerge'],
        )
    )

    fmap = args.map

    if args.p is None:
        print("# No   initial positions are specified, shift on grids")
    else:
        print("# Read initial positions from provided data path {}".format(args.p))
        

    # Convert predicted map to points
    print("# Convert atom probability map to coords")
    device = get_device_names(args.device)[0]
    fca = pjoin(out_dir, "raw_c4.pdb")
    if 'cpu' in device:
        lib_dir = abspath(args.lib)
        print("# getp using CPU")
        ca_success = run_getp(
            fmap, 
            fca, 
            lib_dir=lib_dir, 
            pdb=args.p,
            **getp_args, 
            verbose=True,
        )
    elif 'cuda' in device:
        gpu_id = int(device.replace("cuda:", ""))
        print("# getp using GPU ID = {}".format(gpu_id))
        ca_success = getp(
            fmap, 
            fca, 
            pdb=args.p,
            **getp_args, 
            device=gpu_id,
        )

    if not ca_success:
        raise RuntimeError("Cannot convert map to points")
 
    te = time.time()
    #print("# Time consuming {:.4f}".format(te-ts))


if __name__ =='__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", "-map", help="Input map")
    parser.add_argument("--device", default="cpu", help="Device to run on, CPU or GPU ID")
    # If input coords
    parser.add_argument("--p", "-p", help="Input coords in pdb format")
    # Others
    parser.add_argument("--output", "-o", help="Output directory", default='./')
    parser.add_argument("--lib", "-l", help="Lib directory", default=os.path.join(script_dir, ".."))
    # Mean-Shift controls
    parser.add_argument("--thresh", "-thresh", type=float, default=10.0, help="Map threshold")
    parser.add_argument("--res", "-res", type=float, default=6.0, help="Resolution")
    parser.add_argument("--nt", "-nt", type=int, default=4, help="Num of threads")
    parser.add_argument("--filter", "-filter", type=float, default=0.0, help="Filter thresh")
    parser.add_argument("--dmerge", "-dmerge", type=float, default=1.5, help="Merge distance")
    args = parser.parse_args()
    main(args)
