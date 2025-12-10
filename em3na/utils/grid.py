import numpy as np

def get_aa_type(grid : np.ndarray, coord : np.ndarray, origin=None):
    # AGCU/T
    if origin is not None:
        coord = np.subtract(coord, origin)
    x, y, z = [int(x) for x in coord]
    n0, n1, n2 = grid.shape
    if x >= n2 or x < 0 or \
       y >= n1 or y < 0 or \
       z >= n0 or z < 0:
        return 3
    else:
        # Vote for 3**3 voxels
        votes = [0] * 4
        for xx in range(x-1, x+2):
            for yy in range(y-1, y+2):
                for zz in range(z-1, z+2):
                    if 0 <= xx < n2 and 0 <= yy < n1 and 0 <= zz < n0:
                        v = int(grid[zz, yy, xx])
                        if v < 4:
                            votes[v] += 1
        return np.argmax(votes)

