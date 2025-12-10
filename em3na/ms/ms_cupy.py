from dataclasses import dataclass

import numpy as np
import cupy as cp
from scipy.spatial import cKDTree


@dataclass
class MRC:
    grid: np.ndarray
    origin: np.ndarray
    voxel_size: np.ndarray
    # origin = ori + voxel_size * nxyzstart
    ori: None | np.ndarray = None
    nxyzstart: None | np.ndarray = None

@dataclass
class Point:
    coord: np.ndarray
    density: np.ndarray


_SHIFT_KERNEL_OPTIMIZED = r'''
extern "C" __global__
void shift_kernel_optimized(
    // density data
    const float* grid, // density data (n0, n1, n2) as 1D array
    const int n0, // dim 0 of density data
    const int n1, // dim 1 of density data
    const int n2, // dim 2 of density data

    // coords of grids
    float* coord,         // (n, 3) shifted coords of grids as 1D array
    float* dens,          // (n, ) density of shifted coords of grids as 1D array
    float* status,        // (n, ) status

    // other constants
    const float fmaxd, 
    const float k, 
    const float rshift2,
    const int cnt, 
    const int max_iter
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cnt) return;

    float3 pos0 = {coord[i * 3], coord[i * 3 + 1], coord[i * 3 + 2]};
    float3 pos = pos0;

    int nshift = 0;
    int converged = 0;
    float dtotal = 0.0;

    float statu = 0.0;

    while (nshift < max_iter && !converged){
        nshift++;

        // calculate boundaries
        int stp_x = max(__float2int_rd(pos.x - fmaxd), 0);
        int stp_y = max(__float2int_rd(pos.y - fmaxd), 0);
        int stp_z = max(__float2int_rd(pos.z - fmaxd), 0);
        
        int endp_x = min(__float2int_ru(pos.x + fmaxd + 1), n2);
        int endp_y = min(__float2int_ru(pos.y + fmaxd + 1), n1);
        int endp_z = min(__float2int_ru(pos.z + fmaxd + 1), n0);
       
        // each round reset dtotal
        dtotal = 0.0;
        float3 pos2 = {0.0, 0.0, 0.0};
       
        // optimized loops: change of loop order for better data locality
        for (int zp = stp_z; zp < endp_z; zp++) {
            float rz = (zp - pos.z) * (zp - pos.z);
            
            for (int yp = stp_y; yp < endp_y; yp++) {
                float ry = (yp - pos.y) * (yp - pos.y);
                
                for (int xp = stp_x; xp < endp_x; xp++) {
                    float rx = (xp - pos.x) * (xp - pos.x);
                    float d2 = rx + ry + rz;
        
                    // index to get volume[zp, yp, xp]
                    int index = zp * n1 * n2 + yp * n2 + xp;
            
                    // zyx
                    // 0 0 1 -> 1
                    // 1 0 0 -> n1 * n2
                    // 0 1 0 -> n2

                    float density_val = grid[index];

                    float v = exp(k * d2) * density_val;
 
                    dtotal += v;
                    pos2.x += v * xp;
                    pos2.y += v * yp;
                    pos2.z += v * zp;
                }
            }
        }
       
        if (dtotal < 1e-6) break;
      
        // update position
        pos2.x /= dtotal;
        pos2.y /= dtotal;
        pos2.z /= dtotal;

        // convergence check
        float dx = pos.x - pos2.x;
        float dy = pos.y - pos2.y;
        float dz = pos.z - pos2.z;
        double delta2 = dx * dx + dy * dy + dz * dz;
       
        if (delta2 < 1e-2) converged = 1;

        // max shift check
        float tdx = pos0.x - pos2.x;
        float tdy = pos0.y - pos2.y;
        float tdz = pos0.z - pos2.z;
        float total_delta2 = tdx * tdx + tdy * tdy + tdz * tdz;
        
        if (total_delta2 > rshift2) break;
        
        pos = pos2;
    }

    // write result
    coord[i * 3] = pos.x;
    coord[i * 3 + 1] = pos.y;
    coord[i * 3 + 2] = pos.z;
    dens[i] = dtotal;
    status[i] = nshift;
}
'''

_shift_kernel_optimized = cp.RawKernel(_SHIFT_KERNEL_OPTIMIZED, 'shift_kernel_optimized')

def get_dreso(resol):
    if resol <= 4.0:
        return 2.0
    elif resol <= 5.0:
        return 2.5
    elif resol <= 6.0:
        return 3.0
    elif resol <= 7.0:
        return 3.5
    elif resol <= 8.0:
        return 4.0
    else:
        return 4.5

def ms_gpu_optimized(
    grid,
    origin=np.zeros(3, dtype=np.float32),
    voxel_size=np.zeros(3, dtype=np.float32),
    resol=6.0,
    max_shift=10.0,
    threshold=1e-3,
    # Initial coord specified
    init_coord=None,
):
    n0, n1, n2 = grid.shape
    
    # copy to GPU
    grid_gpu = cp.asarray(
        grid, 
        dtype=cp.float32, 
    )

    if init_coord is None: 
        print("# No initial coord")
        print("# Using contour level of {:.6f}".format(threshold))
        mask = grid_gpu > threshold
        coord = cp.argwhere(mask)
        coord = cp.flip(coord, axis=-1) # flip to xyz space
    else:
        print("# Use initial coord")
        coord = (init_coord - origin) / voxel_size # to grid space
        coord = cp.asarray(coord).astype(cp.float32)

    cnt = len(coord)
    grid_flat = grid_gpu.reshape(-1)
    
    print(f"# Grids to be shifted = {cnt}")
    
    # data preparation
    coord_flat = cp.zeros(cnt * 3, dtype=cp.float32)
    coord_flat[0::3] = coord[:, 0].astype(cp.float32) # all x coord
    coord_flat[1::3] = coord[:, 1].astype(cp.float32) # all y coord
    coord_flat[2::3] = coord[:, 2].astype(cp.float32) # all z coord

    dens_gpu = cp.zeros(cnt, dtype=cp.float32)
    shift = cp.zeros(cnt, dtype=cp.float32)

    # params calculation
    dreso = get_dreso(resol)
    gstep = voxel_size[0]

    fs = (dreso / gstep) * 0.5
    fs = fs * fs
    k = -1.5 / fs
    fmaxd = (dreso / gstep) * 2.0
    rshift2 = (max_shift / gstep) ** 2

    # execute
    threads_per_block = 512  # smaller is faster?
    blocks_per_grid = (cnt + threads_per_block - 1) // threads_per_block
   
    _shift_kernel_optimized(
        (blocks_per_grid,), 
        (threads_per_block,),
        (
            grid_flat,
            n0, 
            n1, 
            n2,

            coord_flat, 
            dens_gpu, 
            shift,

            np.float32(fmaxd), 
            np.float32(k), 
            np.float32(rshift2), 
            cnt, 
            5000,
        ),
    )

    cp.cuda.Stream.null.synchronize()
    
    # update result
    coord = cp.asnumpy(coord_flat.reshape(cnt, 3)) * voxel_size + origin
    dens = cp.asnumpy(dens_gpu)
    
    return coord, dens

def merge(coord, dens, d=1.0):
    print("# Merge distance = {:.4f}".format(d))

    n = len(coord)
    if n == 0:
        return
 
    # construct tree
    tree = cKDTree(coord)
    
    sorted_indices = np.argsort(dens)[::-1]
    kept = np.ones(n, dtype=bool)
   
    n_round = 0 
    for i in sorted_indices:
        if not kept[i]:
            continue
            
        neighbors = tree.query_ball_point([coord[i]], r=d)[0]
        neighbors = np.asarray(neighbors).astype(np.int32)
        neighbors = neighbors[neighbors != i]

        kept[neighbors] = False

        n_round += 1

    return coord[kept], dens[kept]
 
