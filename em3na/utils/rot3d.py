import torch
import torch.nn.functional as F
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

from em3na.utils.cryo_utils import parse_map, write_map

def get_atoms(filename):
    new_lines = []
    ca_coords = []
    coords = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("ATOM"):
                coord = [float(x) for x in [line[i:i + 8] for i in [30, 38, 46]]]
                if line[12:16] == " C4'":
                    ca_coords.append(coord)
                coords.append(coord)
                new_lines.append(line[:-1])
    coords = np.asarray(coords, dtype=np.float32)
    ca_coords = np.asarray(ca_coords, dtype=np.float32)

    return ca_coords, coords, new_lines

def rotate_3d_around_point(image, center, R):
    B, C, D, H, W = image.shape
    device = image.device

    d, h, w = torch.meshgrid(
        torch.linspace(-1, 1, D, device=device),
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    
    grid = torch.stack((w, h, d, torch.ones_like(w)), dim=-1)  # (D, H, W, 4)
    grid = grid.view(-1, 4).T  # (4, N)

    grid[0] = (grid[0] + 1) * (W - 1) / 2
    grid[1] = (grid[1] + 1) * (H - 1) / 2
    grid[2] = (grid[2] + 1) * (D - 1) / 2

    gridx = grid.clone()
    gridx[:3] -= center.clone().view(3, 1)

    grid_rot = R.T @ gridx

    grid_rot[:3] += center.clone().view(3, 1)

    #print((grid.round().int() == grid_rot.round().int()).sum())
    #print((grid.round().int() != grid_rot.round().int()).sum())

    grid_rot[0] = 2 * grid_rot[0] / (W - 1) - 1
    grid_rot[1] = 2 * grid_rot[1] / (H - 1) - 1
    grid_rot[2] = 2 * grid_rot[2] / (D - 1) - 1

    grid_rot = grid_rot.T[:, :3].view(1, D, H, W, 3)
    grid_rot = grid_rot.expand(B, -1, -1, -1, -1)

    rotated_image = F.grid_sample(image, grid_rot, mode='bilinear', padding_mode='zeros', align_corners=True)

    return rotated_image


def random_rotation_matrix():
    quat = np.random.randn(4)
    quat /= np.linalg.norm(quat)
    rot = R.from_quat(quat)
    matrix = rot.as_matrix()
    return matrix

if __name__ == '__main__':
    data, origin, nxyz, voxel_size = parse_map(sys.argv[1], False, None)

    ca_coords, coords, lines= get_atoms(sys.argv[2])

    center = ca_coords.mean(axis=0)

    print(center)

    R = random_rotation_matrix()    

   
    # rotate map 
    #centerx = center[::-1].copy()
    centerx = center
    Rx = torch.eye(4)
    Rx[:3, :3] = torch.from_numpy(R)

    data = rotate_3d_around_point(
        torch.from_numpy(data)[None, None, ...], 
        (torch.from_numpy(centerx) - origin) / voxel_size, 
        Rx, 
    )
    data = data.numpy()[0][0]

    # rotate structrue
    rot_coords = (coords - center[None, ...]) @ R.T + center[None, ...]

    write_map("datax.mrc", data, voxel_size, origin)

    for k, line in enumerate(lines):
        new_line = line[:30] + "{:8.3f}{:8.3f}{:8.3f}".format(rot_coords[k, 0], rot_coords[k, 1], rot_coords[k, 2]) + line[54:]
        print(new_line)

