"""
Utilities for calculating all atom representations.

Code adapted from
- https://github.com/jasonkyuyim/se3_diffusion/blob/master/data/all_atom.py
- https://github.com/Profluent-Internships/MMDiff/blob/main/src/data/components/pdb/all_atom.py
"""

import torch
import torch.nn.functional as F

from em3na.na_utils.data import nucleotide_constants
from em3na.na_utils.data import rigid_utils as ru
from em3na.na_utils.data import vocabulary
from em3na.na_utils.data.complex_constants import NUM_NA_TORSIONS, NUM_PROT_NA_TORSIONS

IDEALIZED_NA_ATOM_POS27 = torch.tensor(nucleotide_constants.nttype_atom27_rigid_group_positions)
IDEALIZED_NA_ATOM_POS27_MASK = torch.any(IDEALIZED_NA_ATOM_POS27, axis=-1)
IDEALIZED_NA_ATOM_POS23 = torch.tensor(
    nucleotide_constants.nttype_compact_atom_rigid_group_positions
)
DEFAULT_NA_RESIDUE_FRAMES = torch.tensor(nucleotide_constants.nttype_rigid_group_default_frame)
NA_RESIDUE_ATOM23_MASK = torch.tensor(nucleotide_constants.nttype_compact_atom_mask)
NA_RESIDUE_GROUP_IDX = torch.tensor(nucleotide_constants.nttype_compact_atom_to_rigid_group)

def create_rna_rigid(rots, trans):
    # takes in separate rotations and translations and returns a unified rigid_utils.Rigid(...) object
    rots = ru.Rotation(rot_mats=rots)
    return ru.Rigid(rots=rots, trans=trans)


def to_atom37_rna(trans, rots, is_na_residue_mask, torsions=None):
    """
    Params:
        trans (tensor) : tensor representing translations
        rots (tensor) : tensor representing rotations
        is_na_residue_mask (tensor) : one-hot nucleotide mask
        torsions (tensor) : predicted/ground truth torsion angles to impute non-frame backbone atoms

    Remarks:
        Takes RNA rots+trans (ie, a RNA frame) and converts it to the sparse ATOM37 representation
    """

    final_atom37 = compute_backbone(
        bb_rigids=create_rna_rigid(rots, trans),
        torsions=torsions, # 16 float values -> NUM_NA_TORSIONS * 2 (each angle is in SO(2))
        is_na_residue_mask=is_na_residue_mask,
    )[0]
    return final_atom37

def to_atom37_rna_all_atom(trans, rots, is_na_residue_mask, torsions=None, aatype=None):
    final_atom37 = compute_all_atom(
        bb_rigids=create_rna_rigid(rots, trans),
        torsions=torsions, 
        is_na_residue_mask=is_na_residue_mask,
        aatype=aatype, 
    )[0]
    return final_atom37


def transrot_to_atom37_rna(transrot_traj, is_na_residue_mask, torsions):
    """
    Params:
        transrot_traj (list) : list of evolving ATOM37 tensors representing atomic coordinates for N_T flow matching timesteps
        is_na_residue_mask (tensor) : one-hot nucleotide mask
        torsions (tensor) : predicted torsion angles from the AngleResNet

    Returns:
        All-atom backbone trajectories with all non-frame atoms placed according to predicted torsion angles
    """
    atom37_traj = []

    for trans, rots in transrot_traj:
        rigids = create_rna_rigid(rots, trans)
        rna_atom37 = compute_backbone(
                bb_rigids=rigids,
                torsions=torsions, # 16 -> NUM_NA_TORSIONS * 2
                is_na_residue_mask=is_na_residue_mask,
            )[0]
        rna_atom37 = rna_atom37.detach().cpu()
        atom37_traj.append(rna_atom37)

    return atom37_traj

def of_na_torsion_angles_to_frames(r, alpha, aatype, rrgdf):
    """
    Params:
        r (tensor) : tensor of rigid_utils.Rigid(...) objects representing RNA Frames
        alpha (tensor) : predicted torsion angles from AngleResNet
        aatype (tensor) : tensor of residue types [ignored]

    Remarks:
        Uses the predicted RNA Frames `r` and provided torsion angles `alpha`
        to impute non-frame backbone atoms into the ATOM37 representation.
    
    Returns:
        A torch tensor in ATOM37 format containing all 13 RNA backbone atoms.
    """
    # [*, N, 11, 4, 4]
    default_4x4 = rrgdf[aatype, ...]

    # [*, N, 11] transformations, i.e.
    #   One [*, N, 11, 3, 3] rotation matrix and
    #   One [*, N, 11, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot1 = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot1[..., 1] = 1

    alpha = torch.cat(
        [
            bb_rot1.expand(*alpha.shape[:-2], -1, -1),
            alpha,
        ],
        dim=-2,
    )

    # [*, N, 11, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1  # The upper-left diagonal value for 3D rotations
    all_rots[..., 1, 1] = alpha[..., 1]  # The first sine angle for 3D rotations
    all_rots[..., 1, 2] = -alpha[..., 0]  # The first cosine angle for 3D rotations
    all_rots[..., 2, 1:] = alpha  # The remaining sine and cosine angles for 3D rotations

    all_rots = ru.Rigid(ru.Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    backbone2_atom1_frame = all_frames[..., 1]  # C2'
    backbone2_atom2_frame = all_frames[..., 2]  # C1'
    backbone2_atom3_frame = all_frames[..., 3]  # N9/N1
    delta_frame_to_frame = all_frames[..., 4]  # O3'
    gamma_frame_to_frame = all_frames[..., 5]  # O5'
    beta_frame_to_frame = all_frames[..., 6]  # P
    alpha1_frame_to_frame = all_frames[..., 7]  # OP1
    alpha2_frame_to_frame = all_frames[..., 8]  # OP2
    tm_frame_to_frame = all_frames[..., 9]  # O2'
    chi_frame_to_frame = all_frames[..., 10]  # N1, N3, N6, N7, C2, C4, C5, C6, and C8

    backbone2_atom1_frame_to_bb = backbone2_atom1_frame
    backbone2_atom2_frame_to_bb = backbone2_atom2_frame
    # note: N9/N1 is built off the relative position of C1'
    backbone2_atom3_frame_to_bb = backbone2_atom2_frame.compose(backbone2_atom3_frame)
    delta_frame_to_bb = delta_frame_to_frame
    gamma_frame_to_bb = gamma_frame_to_frame
    beta_frame_to_bb = gamma_frame_to_bb.compose(beta_frame_to_frame)
    alpha1_frame_to_bb = beta_frame_to_bb.compose(alpha1_frame_to_frame)
    alpha2_frame_to_bb = beta_frame_to_bb.compose(alpha2_frame_to_frame)
    # use `backbone2_atom1/3_frames` to compose `tm` and `chi` frames,
    # since `backbone2_atom1/3_frames` place the `C2'` and `N9/N1` atoms
    # (i.e., two of the second backbone group's atoms)
    tm_frame_to_bb = backbone2_atom1_frame_to_bb.compose(tm_frame_to_frame)
    chi_frame_to_bb = backbone2_atom3_frame_to_bb.compose(chi_frame_to_frame)

    all_frames_to_bb = ru.Rigid.cat(
        [
            all_frames[..., 0].unsqueeze(-1),
            backbone2_atom1_frame_to_bb.unsqueeze(-1),
            backbone2_atom2_frame_to_bb.unsqueeze(-1),
            backbone2_atom3_frame_to_bb.unsqueeze(-1),
            delta_frame_to_bb.unsqueeze(-1),
            gamma_frame_to_bb.unsqueeze(-1),
            beta_frame_to_bb.unsqueeze(-1),
            alpha1_frame_to_bb.unsqueeze(-1),
            alpha2_frame_to_bb.unsqueeze(-1),
            tm_frame_to_bb.unsqueeze(-1),
            chi_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)
    return all_frames_to_global

def na_frames_to_atom23_pos(r, aatype):
    """
    Params:
        r: All rigid groups. [..., N, 11, 3]
        aatype: Residue types. [..., N]

    Remarks:
        Convert nucleic acid (NA) frames to their idealized all atom representation

    Returns:
        Idealized all-atom backbone in ATOM37 format as a torch tensor
    """

    # [*, N, 23]
    group_mask = NA_RESIDUE_GROUP_IDX.to(r.device)[aatype, ...]

    # [*, N, 23, 11]
    group_mask = torch.nn.functional.one_hot(
        group_mask,
        num_classes=DEFAULT_NA_RESIDUE_FRAMES.shape[-3],
    ).to(r.device)

    # [*, N, 23, 11]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 23]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # [*, N, 23, 1]
    frame_atom_mask = NA_RESIDUE_ATOM23_MASK.to(r.device)[aatype, ...].unsqueeze(-1)

    # [*, N, 23, 3]
    frame_null_pos = IDEALIZED_NA_ATOM_POS23.to(r.device)[aatype, ...]

    pred_positions = t_atoms_to_global.apply(frame_null_pos)
    pred_positions = pred_positions * frame_atom_mask

    return pred_positions

def na_frames_to_rigid_groups(r, aatype, atom_pos):
    # [*, N, 23]
    group_mask = NA_RESIDUE_GROUP_IDX.to(r.device)[aatype, ...]

    # [*, N, 23, 11]
    group_mask = torch.nn.functional.one_hot(
        group_mask,
        num_classes=DEFAULT_NA_RESIDUE_FRAMES.shape[-3],
    ).to(r.device)

    # [*, N, 23, 11]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 23]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # [*, N, 23, 1]
    frame_atom_mask = NA_RESIDUE_ATOM23_MASK.to(r.device)[aatype, ...].unsqueeze(-1)

    # [*, N, 23, 3]
    frame_null_pos = IDEALIZED_NA_ATOM_POS23.to(r.device)[aatype, ...]
    pred_positions = t_atoms_to_global.invert_apply(atom_pos)
    pred_positions = pred_positions * frame_atom_mask

    return pred_positions

def compute_backbone(bb_rigids, torsions, is_na_residue_mask, aatype=None):
    """
    Params:
        bb_rigids (tensor) : torch tensor of Frame objects comprising rotations and translations
        torsions : predicted torsion angles from AngleResNet
        is_na_residue_mask (tensor) : one-hot nucleotide mask
        aatype (tensor) : tensor representing residue types [ignored]

    Remarks:
        This method takes in frame objects and converts them into all-atom RNA backbones.

    Returns:
        All-atom RNA backbones with frame atoms and non-frame atoms imputed according to torsion angles.
        Stored in ATOM37 format as a torch tensor.
    """

    na_inputs_present = is_na_residue_mask.any().item()
    torsions = torsions.view(torsions.shape[0], torsions.shape[1], NUM_NA_TORSIONS*2) # NOTE: reshape torsion tensor
    
    torsion_angles = torch.tile(
        torsions[..., None, :2],
        tuple([1 for _ in range(len(bb_rigids.shape))]) + (NUM_PROT_NA_TORSIONS, 1),
    )

    if na_inputs_present:
        """
        NOTE:
        For nucleic acid molecules, we insert their predicted torsion angles
        as the first eight torsion entries and tile the remaining torsion entries
        using the first of their eight predicted torsion angles

        Even though the `atom27_to_frames()` method in data_transforms.py uses NUM_PROT_NA_TORSIONS,
        the 8 torsion angles for RNA backbones are extracted from them using array slicing above:
        """
        masked_angles = torsions.view(torsions.shape[0], -1, 8, 2)
        torsion_angles[..., :NUM_NA_TORSIONS, :] = masked_angles
    
    rot_dim = ((4,) if bb_rigids._rots._quats is not None else (3, 3)) # i.e., otherwise, anticipate rotations being used
    
    # fill in residue types for RNA [ACGU]
    aatype = (aatype if aatype is not None else torch.zeros(bb_rigids.shape, device=bb_rigids.device, dtype=torch.long))
    
    if na_inputs_present:
        na_bb_rigids = bb_rigids[is_na_residue_mask].reshape(
            new_rots_shape=torch.Size((bb_rigids.shape[0], -1, *rot_dim)),
            new_trans_shape=torch.Size((bb_rigids.shape[0], -1, 3)),
        )
        
        na_torsion_angles = torsion_angles.view(torsion_angles.shape[0], -1, NUM_PROT_NA_TORSIONS, 2)
        
        na_aatype = aatype[is_na_residue_mask].view(aatype.shape[0], -1)
        na_aatype_in_original_range = na_aatype.min() > vocabulary.protein_restype_num
        
        effective_na_aatype = (
            na_aatype - (vocabulary.protein_restype_num + 1)
            if na_aatype_in_original_range
            else na_aatype
        )

        all_na_frames = of_na_torsion_angles_to_frames(
            r=na_bb_rigids,
            alpha=na_torsion_angles,
            aatype=effective_na_aatype,
            rrgdf=DEFAULT_NA_RESIDUE_FRAMES.to(bb_rigids.device),
        )

        na_atom23_pos = na_frames_to_atom23_pos(all_na_frames, effective_na_aatype)
        
    if na_inputs_present:
        atom23_pos = na_atom23_pos
    else:
        raise Exception("Either protein or nucleic acid chains must be provided as inputs.")
    
    atom37_bb_pos = torch.zeros(bb_rigids.shape + (37, 3), device=bb_rigids.device)
    atom37_bb_supervised_mask = torch.zeros(bb_rigids.shape + (37,), device=bb_rigids.device, dtype=torch.bool)

    # NOTE: map non-frame atoms into correct indexes in ATOM37 representation
    if na_inputs_present:
        # note: nucleic acids' atom23 bb order = ['C3'', 'C4'', 'O4'', 'C2'', 'C1'', 'C5'', 'O3'', 'O5'', 'P', 'OP1', 'OP2', 'N9', ...]
        # note: nucleic acids' atom37 bb order = ['C1'', 'C2'', 'C3'', 'C4'', 'C5'', '05'', '04'', 'O3'', 'O2'', 'P', 'OP1', 'OP2', 'N1', 'N2', 'N3', 'N4', 'N6', 'N7', 'N9', ...]
        #                                                         2      3      4             6      7            9    10     11     12
        atom37_bb_pos[..., 2:4, :][is_na_residue_mask] = atom23_pos[..., :2, :][is_na_residue_mask]  # reindex C3' and C4'
        atom37_bb_pos[..., 6, :][is_na_residue_mask] = atom23_pos[..., 2, :][is_na_residue_mask]  # reindex O4'
        atom37_bb_pos[..., 1, :][is_na_residue_mask] = atom23_pos[..., 3, :][is_na_residue_mask]  # reindex C2'
        atom37_bb_pos[..., 0, :][is_na_residue_mask] = atom23_pos[..., 4, :][is_na_residue_mask]  # reindex C1'
        atom37_bb_pos[..., 4, :][is_na_residue_mask] = atom23_pos[..., 5, :][is_na_residue_mask]  # reindex C5'
        atom37_bb_pos[..., 7, :][is_na_residue_mask] = atom23_pos[..., 6, :][is_na_residue_mask]  # reindex O3'
        atom37_bb_pos[..., 5, :][is_na_residue_mask] = atom23_pos[..., 7, :][is_na_residue_mask]  # reindex O5'
        atom37_bb_pos[..., 9:12, :][is_na_residue_mask] = atom23_pos[..., 8:11, :][is_na_residue_mask]  # reindex P, OP1, and OP2
        atom37_bb_pos[..., 18, :][is_na_residue_mask] = atom23_pos[..., 11, :][is_na_residue_mask]  # reindex N9, which instead reindexes N1 for pyrimidine residues
        
        atom37_bb_supervised_mask[..., :8][is_na_residue_mask] = True
        atom37_bb_supervised_mask[..., 9:12][is_na_residue_mask] = True
        atom37_bb_supervised_mask[..., 18][is_na_residue_mask] = True

    atom37_mask = torch.any(atom37_bb_pos, axis=-1)

    return atom37_bb_pos, atom37_mask, atom37_bb_supervised_mask, aatype, atom23_pos


def compute_all_atom(bb_rigids, torsions, is_na_residue_mask, aatype=None):
    na_inputs_present = is_na_residue_mask.any().item()
    torsions = torsions.view(torsions.shape[0], torsions.shape[1], 10 * 2) # NOTE: reshape torsion tensor
    
    torsion_angles = torch.tile(
        torsions[..., None, :2],
        tuple([1 for _ in range(len(bb_rigids.shape))]) + (10, 1),
    )

    if na_inputs_present:
        masked_angles = torsions.view(torsions.shape[0], -1, 10, 2)
        torsion_angles[..., :10, :] = masked_angles
    
    rot_dim = ((4,) if bb_rigids._rots._quats is not None else (3, 3)) # i.e., otherwise, anticipate rotations being used
    
    # fill in residue types for RNA [ACGU]
    aatype = (aatype if aatype is not None else torch.zeros(bb_rigids.shape, device=bb_rigids.device, dtype=torch.long))
    
    if na_inputs_present:
        na_bb_rigids = bb_rigids[is_na_residue_mask].reshape(
            new_rots_shape=torch.Size((bb_rigids.shape[0], -1, *rot_dim)),
            new_trans_shape=torch.Size((bb_rigids.shape[0], -1, 3)),
        )
        
        na_torsion_angles = torsion_angles.view(torsion_angles.shape[0], -1, 10, 2)

        na_aatype = aatype[is_na_residue_mask].view(aatype.shape[0], -1)
        na_aatype_in_original_range = na_aatype.min() > vocabulary.protein_restype_num
    
        """
        effective_na_aatype = (
            na_aatype - (vocabulary.protein_restype_num + 1)
            if na_aatype_in_original_range
            else na_aatype
        )
        """

        # DA DC DG DT  A  C  G  U
        # 21 22 23 24 25 26 27 28
        assert ((21 <= na_aatype) & (29 > na_aatype)).all()

        # 0  1  2  3  4  5  6  7
        effective_na_aatype = na_aatype - 25 + 4

        all_na_frames = of_na_torsion_angles_to_frames(
            r=na_bb_rigids,
            alpha=na_torsion_angles,
            aatype=effective_na_aatype,
            rrgdf=DEFAULT_NA_RESIDUE_FRAMES.to(bb_rigids.device),
        )

        na_atom23_pos = na_frames_to_atom23_pos(all_na_frames, effective_na_aatype)

        # MOD return
        #return (na_atom23_pos, )

    if na_inputs_present:
        atom23_pos = na_atom23_pos
    else:
        raise Exception("Either protein or nucleic acid chains must be provided as inputs.")
    
    atom37_pos = torch.zeros(bb_rigids.shape + (37, 3), device=bb_rigids.device)
    atom37_bb_supervised_mask = torch.zeros(bb_rigids.shape + (37,), device=bb_rigids.device, dtype=torch.bool)

    # NOTE: map non-frame atoms into correct indexes in ATOM37 representation
    if na_inputs_present:
        # from restype_name_to_compact_atom_names
        # note: nucleic acids' atom23 bb order = 
        # ['C3'', 'C4'', 'O4'', 'C2'', 'C1'', 'C5'', 'O3'', 'O5'', 'P', 'OP1', 'OP2', 'N9', ...]

        # from atom_types
        # note: nucleic acids' atom37 bb order = 
        # ['C1'', 'C2'', 'C3'', 'C4'', 'C5'', '05'', '04'', 'O3'', 'O2'', 'P', 'OP1', 'OP2', 'N1', 'N2', 'N3', 'N4', 'N6', 'N7', 'N9', ...]
        #                                                         2      3      4             6      7            9    10     11     12
        atom37_pos[..., 2:4, :][is_na_residue_mask] = atom23_pos[..., :2, :][is_na_residue_mask]  # reindex C3' and C4'
        atom37_pos[..., 6, :][is_na_residue_mask] = atom23_pos[..., 2, :][is_na_residue_mask]  # reindex O4'
        atom37_pos[..., 1, :][is_na_residue_mask] = atom23_pos[..., 3, :][is_na_residue_mask]  # reindex C2'
        atom37_pos[..., 0, :][is_na_residue_mask] = atom23_pos[..., 4, :][is_na_residue_mask]  # reindex C1'
        atom37_pos[..., 4, :][is_na_residue_mask] = atom23_pos[..., 5, :][is_na_residue_mask]  # reindex C5'
        atom37_pos[..., 7, :][is_na_residue_mask] = atom23_pos[..., 6, :][is_na_residue_mask]  # reindex O3'
        atom37_pos[..., 5, :][is_na_residue_mask] = atom23_pos[..., 7, :][is_na_residue_mask]  # reindex O5'
        atom37_pos[..., 9:12, :][is_na_residue_mask] = atom23_pos[..., 8:11, :][is_na_residue_mask]  # reindex P, OP1, and OP2
        atom37_pos[..., 18, :][is_na_residue_mask] = atom23_pos[..., 11, :][is_na_residue_mask]  # reindex N9, which instead reindexes N1 for pyrimidine residues
        
        # MOD add base atoms

        # atom37
        # 02 N1 N2 N3 N4 N6 N7 N9 C2 C4 C5 C6 C8 O2 O4 O6
        #  8 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26

        ###############
        ### For RNA ###
        ###############
        # atom23
        # 11 12 13 14 15 16 17 18 19 20 21 22
        # A
        # N9 02 C4 N1 N3 N6 N7 C2 C5 C6 C8
        # C
        # N1 02 C2 N3 N4 C4 C5 C6 O2
        # G
        # N9 02 C4 N1 N2 N3 N7 C2 C5 C6 C8 O6
        # U
        # N1 02 C2 N3 C4 C5 C6 O2 O4

        A_mask = (effective_na_aatype == 4)
        C_mask = (effective_na_aatype == 5)
        G_mask = (effective_na_aatype == 6)
        U_mask = (effective_na_aatype == 7)

        x = atom37_pos[A_mask]
        x[..., [18, 8, 20, 12, 14, 16, 17, 19, 21, 22, 23], :] = atom23_pos[A_mask][..., list(range(11, 22)), :]
        atom37_pos[A_mask] = x

        x = atom37_pos[C_mask]
        x[..., [12, 8, 19, 14, 15, 20, 21, 22, 24], :] = atom23_pos[C_mask][..., list(range(11, 20)), :]
        atom37_pos[C_mask] = x

        x = atom37_pos[G_mask]
        x[..., [18, 8, 20, 12, 13, 14, 17, 19, 21, 22, 23, 26], :] = atom23_pos[G_mask][..., list(range(11, 23)), :]
        atom37_pos[G_mask] = x

        x = atom37_pos[U_mask]
        x[..., [12, 8, 19, 14, 20, 21, 22, 24, 25], :] = atom23_pos[U_mask][..., list(range(11, 20)), :]
        atom37_pos[U_mask] = x


        ###############
        ### For DNA ###
        ###############
        # atom23 no O2' (02)
        # 11 12 13 14 15 16 17 18 19 20 21 22
        # DA
        # N9 C4 N1 N3 N6 N7 C2 C5 C6 C8
        # DC
        # N1 C2 N3 N4 C4 C5 C6 O2
        # DG
        # N9 C4 N1 N2 N3 N7 C2 C5 C6 C8 O6
        # DU
        # N1 C2 N3 C4 C5 C6 O2 O4
        A_mask = (effective_na_aatype == 0)
        C_mask = (effective_na_aatype == 1)
        G_mask = (effective_na_aatype == 2)
        U_mask = (effective_na_aatype == 3)

        x = atom37_pos[A_mask]
        x[..., [18, 20, 12, 14, 16, 17, 19, 21, 22, 23], :] = atom23_pos[A_mask][..., list(range(11, 21)), :]
        atom37_pos[A_mask] = x

        x = atom37_pos[C_mask]
        x[..., [12, 19, 14, 15, 20, 21, 22, 24], :] = atom23_pos[C_mask][..., list(range(11, 19)), :]
        atom37_pos[C_mask] = x

        x = atom37_pos[G_mask]
        x[..., [18, 20, 12, 13, 14, 17, 19, 21, 22, 23, 26], :] = atom23_pos[G_mask][..., list(range(11, 22)), :]
        atom37_pos[G_mask] = x

        x = atom37_pos[U_mask]
        x[..., [12, 19, 14, 20, 21, 22, 24, 25], :] = atom23_pos[U_mask][..., list(range(11, 19)), :]
        atom37_pos[U_mask] = x
        

        atom37_bb_supervised_mask[..., :8][is_na_residue_mask] = True
        atom37_bb_supervised_mask[..., 9:12][is_na_residue_mask] = True
        atom37_bb_supervised_mask[..., 18][is_na_residue_mask] = True

    atom37_mask = torch.any(atom37_pos, axis=-1)

    return atom37_pos, atom37_mask, atom37_bb_supervised_mask, aatype, atom23_pos
