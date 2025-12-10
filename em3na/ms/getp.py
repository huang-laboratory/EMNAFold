import numpy as np
import cupy as cp

from em3na.utils.cryo_utils import parse_map
from em3na.ms.ms_cupy import ms_gpu_optimized, merge

def read_pdb(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    coords = []
    for line in lines:
        xyz = [float(x) for x in [line[i:i+8] for i in [30, 38, 46]]]
        coords.append(xyz)
    coords = np.asarray(coords, dtype=np.float32)
    return coords

def write_pdb(filename, coords, dens):
    with open(filename, 'w') as f:
        for i, coord in enumerate(coords):
            n = i + 1
            if n > 9999:
                n = 9999

            f.write("ATOM  {:>5d}  CA  GLY A{:>4d}    {:8.3f}{:8.3f}{:8.3f}{:>6.2f}{:>6.2f}{}\n".format(
                n,
                n,
                coord[0],
                coord[1],
                coord[2],
                dens[i],
                dens[i],
                " " * 15,
            ))
            f.write("TER\n")

def getp(
    map, # map path
    output, # output pdb path
    pdb=None,
    res=6.0, 
    thresh=20, 
    filter=0.0, 
    dmerge=1.0, 
    max_shift=10.0,
    verbose=True,
    device=0,
    **kwargs,
):
    if pdb is not None:
        init_coord = read_pdb(pdb)
    else:
        init_coord = None

    # read map
    grid, origin, _, voxel_size = parse_map(map, False, None, device="cpu")

    # ms
    with cp.cuda.Device(device):
        coord, dens = ms_gpu_optimized(
            grid,
            origin=origin,
            voxel_size=voxel_size,
            resol=res,
            max_shift=max_shift,
            threshold=thresh,
            init_coord=init_coord,
        )

    # merge
    new_coord, new_dens = merge(coord, dens, d=dmerge)
    dmin = new_dens.min()
    dmax = new_dens.max()
    new_dens = (new_dens - dmin) / (dmax - dmin + 1e-6)

    # write
    write_pdb(output, new_coord, new_dens)

    return True
