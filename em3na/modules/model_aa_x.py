import torch
import torch.nn as nn
from typing import List
import contextlib

from em3na.modules.cryo_init_x import CryoInit
from em3na.modules.track_aa import TrackBlock, TorsionNet
from em3na.modules.sequence_attention import SequenceAttention

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


def get_rel_pos(seq_idx, min_pos=-32, max_pos=32):
    rel_pos = seq_idx[..., :, None] - seq_idx[..., None, :]  # shape (..., l, l)
    rel_pos = torch.clip(rel_pos, min_pos, max_pos)
    return rel_pos

class FusedAAPredictor(nn.Module):
    def __init__(
        self, 
        d_node: int = 256, 
        n_classes: int = 4,
    ):

        super().__init__()
        self.linear1 = nn.Linear(d_node, d_node)
        self.linear2 = nn.Linear(d_node, d_node)

        self.aa_head = nn.Sequential(
            nn.Linear(d_node, d_node),
            nn.ReLU(),
            nn.Linear(d_node, d_node),
            nn.ReLU(),
            nn.Linear(d_node, n_classes),
        )

    def forward(self, node_density, node):
        x = self.linear1(node_density) + self.linear2(node)
        return self.aa_head(x)

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
        self.n_block = n_block

        self.min_pos = -32
        self.max_pos =  32

        self.edge_embed = nn.Linear(64 + 1 + 2 + 2, d_edge)

        # Feature init
        self.init = CryoInit(
            d_node=d_node,
            d_edge=d_edge,
            d_cryo_emb=d_node,
            k=k,
        )

        # Track blocks
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

        # Seq attention blocks
        self.seq_attn_blocks = nn.ModuleList()
        for i in range(self.n_block):
            self.seq_attn_blocks.append(SequenceAttention(
                d=d_node,
                d_seq=1280,
                d_head=d_head,
                n_head=n_head,
                checkpoint=use_checkpoint,
            ))

        # AA predictor
        self.aa_predictor = FusedAAPredictor(
            d_node=d_node, 
            n_classes=6,
        )

    def forward(
        self, 
        affines: torch.Tensor, 
        seq_embed: torch.Tensor, 
        seq_idx: torch.Tensor,
        chain_idx: torch.Tensor,
        pairing_matrix: torch.Tensor, 
        cryo_grids: List[torch.Tensor],
        cryo_global_origins: List[torch.Tensor],
        cryo_voxel_sizes: List[torch.Tensor], 
        batch=None, 
        **kwargs, 
    ):

        if batch is None:
            batch = torch.zeros( len(affines) ).to(affines.device).long()

        # Embedding edge
        seq_idx = chain_idx * int(1e5) + seq_idx
        seq_idx = seq_idx.long()

        rel_pos = get_rel_pos(seq_idx, self.min_pos, self.max_pos) + self.max_pos
        rel_pos_embed = torch.nn.functional.one_hot(rel_pos.long(), num_classes=64 + 1)

        same_chain = chain_idx[..., None, :] == chain_idx[..., :, None]
        same_chain_embed = torch.nn.functional.one_hot(same_chain.long(), num_classes=2)

        pairing_embed = torch.nn.functional.one_hot(pairing_matrix.long(), num_classes=2)

        #n0, n1 = pairing_embed.shape[:2]
        #if n0 != same_chain_embed.shape[0] or n0 != rel_pos_embed.shape[0]:
        #    print(n0, n1)
        #if n1 != same_chain_embed.shape[1] or n1 != rel_pos_embed.shape[1]:
        #    print(n0, n1)

        # Initial node and edge
        node = 0.0 # (..., l, dn)
        edge = self.edge_embed(
            torch.cat(
                [same_chain_embed, rel_pos_embed, pairing_embed],
                dim=-1,
            ).float(), 
        ) # (..., l, l, de) 

        # Init features from map
        node_density, edge_density, _ = self.init(
            affines=affines,
            cryo_grids=cryo_grids, 
            cryo_global_origins=cryo_global_origins, 
            cryo_voxel_sizes=cryo_voxel_sizes, 
            batch=batch,
        )

        # Agg all feats
        node = node + node_density
        edge = edge + edge_density

        # Three-Track block
        seq_mask = torch.ones_like( seq_embed[..., 0] )

        for i in range(self.n_block):
            # Track blocks
            node, edge, = self.blocks[i](
                node, 
                edge, 
                affines, 
                use_checkpoint=self.use_checkpoint, 
            )

            # Cross-attention
            node = self.seq_attn_blocks[i](
                node, 
                seq_embed, 
                seq_mask,
            )

        # Predict aa type from fused feats
        pred_aatype = self.aa_predictor(node_density, node)

        return Output(
            pred_aatype=pred_aatype, 
        )


import time
if __name__ == '__main__':
    model = Model(n_block=6, use_checkpoint=True).cuda()
    print(sum(p.numel() for p in model.parameters()))

    L = 384
    affines=torch.randn(L, 3, 4).cuda()
    cryo_grids=[torch.zeros(1, 1, 120, 120, 120).cuda()]
    cryo_global_origins=[torch.zeros(3).cuda()]
    cryo_voxel_sizes=[torch.ones(3).cuda()]

    seq_embed = torch.randn(1, 3000, 1280).cuda()

    batch = torch.zeros(L).long().cuda()
    seq_idx = torch.arange(L).long().cuda()
    chain_idx = torch.zeros(L).long().cuda()
    pairing_matrix = torch.zeros(L, L).long().cuda()


    for i in range(20):
        t0 = time.time()

        out = model(
            affines=affines,
            seq_embed=seq_embed, 
            seq_idx=seq_idx,
            chain_idx=chain_idx,
            pairing_matrix=pairing_matrix, 
            cryo_grids=cryo_grids,
            cryo_global_origins=cryo_global_origins,
            cryo_voxel_sizes=cryo_voxel_sizes,
            batch=batch,
        )
        t1 = time.time()

        print("{:.4f}".format(t1 - t0))

        print(out['pred_aatype'].shape)
        print(out['pred_aatype'][0])

        time.sleep(3)


