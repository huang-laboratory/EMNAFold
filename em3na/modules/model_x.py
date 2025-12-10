import torch
import torch.nn as nn
from typing import List
import contextlib

from em3na.modules.cryo_init_x import CryoInit
from em3na.modules.track_x import TrackBlock, TorsionNet

class Output:
    def __init__(self, **kwargs):
        self.data = dict()
        for k, v in kwargs.items():
            self.data[k] = v

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def to(self, device_or_dtype):
        for k, v in self.data.items():
            if torch.is_tensor(v):
                self.data[k] = v.to(device_or_dtype)
            elif isinstance(v, list):
                self.data[k] = [x.to(device_or_dtype) for x in v]
        return self
        

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def update(self, **kwargs):
        self.data.update(kwargs)

    def __repr__(self):
        return f"Output({self.data})"



class Model(nn.Module):
    def __init__(
        self, 
        d_node=256,
        d_edge=128,
        d_bias=64,
        d_head=48,
        n_qk_point=4,
        n_v_point=8, 
        n_head=8,
        n_block=8, 
        k=64,
        p_drop=0.10,
        use_checkpoint=False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Feature init
        self.init = CryoInit(
            d_node=d_node,
            d_edge=d_edge,
            d_cryo_emb=d_node,
            k=k,
        )

        # Track blocks
        self.n_block = n_block
        self.blocks = nn.ModuleList()
        for i in range(self.n_block):
            self.blocks.append(TrackBlock(
                d_node=d_node,
                d_edge=d_edge,
                d_bias=d_bias,
                d_head=d_head,
                n_head=n_head,
                n_qk_point=n_qk_point,
                n_v_point=n_v_point,
                p_drop=p_drop,
            ))

        # Torsions
        self.torsion_predictor = TorsionNet(d_node, d_node, n_tors=10)

        # pLDDT
        self.rmsd_predictor = nn.Sequential(
            nn.Linear(d_node, d_node),
            nn.ReLU(),
            nn.Linear(d_node, d_node),
            nn.ReLU(),
            nn.Linear(d_node, 1), 
        )


    def forward(
        self, 
        affines: torch.Tensor, 
        cryo_grids: List[torch.Tensor],
        cryo_global_origins: List[torch.Tensor],
        cryo_voxel_sizes: List[torch.Tensor], 
        batch=None, 
        run_iters=1, 
        **kwargs, 
    ):

        if batch is None:
            batch = torch.zeros( len(affines) ).to(affines.device).long()

        init_affines = affines

        for run_iter in range(run_iters):
            with torch.no_grad() if run_iter < run_iters - 1 else contextlib.nullcontext():
                # Init features from map
                node, edge, node_mask = self.init(
                    affines=init_affines,
                    cryo_grids=cryo_grids, 
                    cryo_global_origins=cryo_global_origins, 
                    cryo_voxel_sizes=cryo_voxel_sizes, 
                    batch=batch,
                )

                # Three-Track block
                affines_list = []
                for i in range(self.n_block):
                    node, edge, affines = self.blocks[i](
                        node, 
                        edge, 
                        affines, 
                        mask=node_mask, 
                        use_checkpoint=self.use_checkpoint, 
                    )

                    affines_list.append(affines)

                # Torsion prediction
                torsions = self.torsion_predictor(node)

                # pLDDT prediction
                rmsd = self.rmsd_predictor(node)

                # For next iteration
                init_affines = affines_list[-1]

        return Output(
            pred_torsions=torsions, 
            pred_affines=affines_list, 
            pred_positions=[affines[..., :3, -1] for affines in affines_list],
            pred_rmsd=rmsd, 
        )

import time
if __name__ == '__main__':
    model = Model().cuda()
    print(sum(p.numel() for p in model.parameters()))

    for i in range(20):
        t0 = time.time()
        batch = torch.zeros(128).long().cuda()

        out = model(
            affines=torch.randn(128, 3, 4).cuda(),
            cryo_grids=[torch.zeros(1, 1, 120, 120, 120).cuda()],
            cryo_global_origins=[torch.zeros(3).cuda()],
            cryo_voxel_sizes=[torch.ones(3).cuda()],
            batch=batch,
            run_iters=1,
        )
        t1 = time.time()

        print("{:.4f}".format(t1 - t0))

        time.sleep(1)
