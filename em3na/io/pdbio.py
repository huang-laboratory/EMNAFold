import re
import os
import math
from typing import List
from copy import deepcopy
from collections import defaultdict, OrderedDict

import numpy as np
import torch

from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.mmcifio import MMCIFIO, mmcif_order
from Bio.PDB import PDBParser, MMCIFParser

from em3na.na_utils.data import complex_constants as cc
from em3na.na_utils.data import nucleotide_constants as nc
from em3na.na_utils.data import protein_constants as pc


"""
chain_names = []
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
# One-letter chain id
for i in range(len(letters)):
    chain_names.append(letters[i])
# Two-letter chain id
for i in range(len(letters)):
    for k in range(len(letters)):
        chain_names.append(letters[i]+letters[k])
# In total we have 62 + 62 * 62 ~= 4000 chain ids
# Three or more-letter chain ids
for l in range(len(letters)):
    for m in range(len(letters)):
        for n in range(len(letters)):
            chain_names.append(letters[l]+letters[m]+letters[n])
# 62 * 62 * 62 = 238328 chains
"""

# Read chain names at local
with open( os.path.join(os.path.dirname(__file__), "chain_names.txt" ), "r") as f:
    chain_names = f.readline()
    chain_names = chain_names.strip().split("|")


class CIFXIO(MMCIFIO):
    def _save_dict(self, out_file):
        label_seq_id = deepcopy(self.dic["_atom_site.auth_seq_id"])
        auth_seq_id = deepcopy(self.dic["_atom_site.auth_seq_id"])
        self.dic["_atom_site.label_seq_id"] = label_seq_id
        self.dic["_atom_site.auth_seq_id"] = auth_seq_id

        # Adding missing "pdbx_formal_charge", "auth_comp_id", "auth_atom_id" to complete a record
        N = len(self.dic["_atom_site.group_PDB"])
        self.dic["_atom_site.pdbx_formal_charge"] = ["?"]*N
        self.dic["_atom_site.auth_comp_id"] = deepcopy(self.dic["_atom_site.label_comp_id"])
        self.dic["_atom_site.auth_asym_id"] = deepcopy(self.dic["_atom_site.label_asym_id"])
        self.dic["_atom_site.auth_atom_id"] = deepcopy(self.dic["_atom_site.label_atom_id"])

        # Handle an extra space at the end of _atom_site.xxx
        _atom_site = mmcif_order["_atom_site"]
        _atom_site = [x.strip() + " " for x in _atom_site]
        mmcif_order["_atom_site"] = _atom_site

        new_dic = defaultdict()
        for k, v in self.dic.items():
            if k[:11] == "_atom_site.":
                new_k = k.strip() + " "
            else:
                new_k = k
            new_dic[new_k] = v
        self.dic = new_dic

        return super()._save_dict(out_file)

def split_to_chains(chain_idx, *args):
    # split
    n_chain = chain_idx.max() + 1
    ret = ()
    for t in args:
        list_t = []
        for i in range(n_chain):
            mask = chain_idx == i
            list_t.append(t[mask])

        ret += (list_t, )
        
    return ret


def na_make_atom27_to_atom23_index():
    atom27_to_atom23_idx = dict()

    for resname in nc.restypes:
        atoms = nc.restype_name_to_compact_atom_names[resname]
        
        res_atom27_to_atom23_idx = []
        for atom in atoms:
            try:
                idx = nc.atom_types.index(atom)
            except ValueError as e:
                idx = -1

            if -1 <= idx < 27:
                res_atom27_to_atom23_idx.append( idx )

        res_atom27_to_atom23_idx = np.asarray(res_atom27_to_atom23_idx, dtype=np.int32)

        atom27_to_atom23_idx[resname] = res_atom27_to_atom23_idx

    return atom27_to_atom23_idx


def na_make_atom23_to_atom27_index():
    atom23_to_atom27_idx = dict()

    for resname in nc.restypes:
        atoms = nc.restype_name_to_compact_atom_names[resname]
    
        res_atom23_to_atom27_idx = []
        for atom in nc.atom_types:
            try:
                idx = atoms.index(atom)
            except ValueError as e:
                idx = -1

            if -1 <= idx < 23:
                res_atom23_to_atom27_idx.append( idx )

        res_atom23_to_atom27_idx = np.asarray(res_atom23_to_atom27_idx, dtype=np.int32)

        atom23_to_atom27_idx[resname] = res_atom23_to_atom27_idx

    return atom23_to_atom27_idx
    

def extend(pos, n=23):
    if len(pos) < n:
        shape_to_add = (n - len(pos),) + pos.shape[1:]
        return np.concatenate(
            [pos, np.zeros(shape_to_add, dtype=pos.dtype)],
            axis=0
        )
    else:
        return pos


def atom27_to_atom23(atom_pos, atom_mask, res_type):
    new_atom_pos = []
    new_atom_mask = []

    res_to_idx = na_make_atom27_to_atom23_index()

    for k, res in enumerate(res_type):
        resname_1 = cc.restype_index_to_1[res]
        if 21 <= res <= 28:
            resname_1 = resname_1.upper()
        else:
            raise NotImplementedError

        idxs = res_to_idx[resname_1]
        
        new_atom_pos.append(extend(atom_pos[k][idxs], 23))

        atom_mask_x = atom_mask[k][idxs]
        atom_mask_x[idxs == -1] = 0.0

        new_atom_mask.append(extend(atom_mask_x, 23))

    new_atom_pos = np.asarray(new_atom_pos)
    new_atom_mask = np.asarray(new_atom_mask)

    new_atom_pos = new_atom_pos * new_atom_mask[..., None]
        
    return new_atom_pos, new_atom_mask



def atom23_to_atom27(atom_pos, atom_mask, res_type):
    new_atom_pos = []
    new_atom_mask = []

    res_to_idx = na_make_atom23_to_atom27_index()

    for k, res in enumerate(res_type):
        resname_1 = cc.restype_index_to_1[res]

        if 21 <= res <= 28:
            resname_1 = resname_1.upper()
        else:
            raise NotImplementedError

        idxs = res_to_idx[resname_1]
        
        new_atom_pos.append(extend(atom_pos[k][idxs], 27))

        atom_mask_x = atom_mask[k][idxs]
        atom_mask_x[idxs == -1] = 0.0

        new_atom_mask.append(extend(atom_mask_x, 27))

    new_atom_pos = np.asarray(new_atom_pos)
    new_atom_mask = np.asarray(new_atom_mask)

    new_atom_pos = new_atom_pos * new_atom_mask[..., None]
        
    return new_atom_pos, new_atom_mask


def fix_quotes(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        # Handle '' issues
        fixed_lines = [re.sub(r"'([A-Z0-9]+)''", '"\\1\'"', line) for line in lines]
        temp_filename = filename + ".tmp"
        with open(temp_filename, 'w') as f:
            f.writelines(fixed_lines)
        os.replace(temp_filename, filename)
    except IOError as e:
        print(f"Error processing file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def chains_atom_pos_to_pdb(
    filename,
    chains_atom_pos,
    chains_atom_mask,
    chains_res_type,
    chains_res_idx=None,
    chains_idx=None, 
    chains_bfactor=None,
    chains_occupancy=None, 
    suffix='cif',
    remarks=None,
):
    # For different chains
    assert len(chains_atom_pos) == len(chains_atom_mask)
    if chains_occupancy is None:
        chains_occupancy = []
        for k in range(len(chains_atom_pos)):
            chains_occupancy.append( np.full(len(chains_atom_pos[k]), 1.0, dtype=np.float32) )

    if chains_bfactor is None:
        chains_bfactor = []
        for k in range(len(chains_atom_pos)):
            chains_bfactor.append( np.ones_like(chains_atom_mask[k], dtype=np.float32) * 100.0 )

    if chains_res_type is None:
        chains_res_type = []
        for k in range(len(chains_atom_pos)):
            chains_res_type.append( np.full(len(chains_atom_pos[k]), 0, dtype=np.int32) )

    if chains_res_idx is None:
        chains_res_idx = []
        for k in range(len(chains_atom_pos)):
            chains_res_idx.append( np.arange(len(chains_atom_pos[k]), dtype=np.int32) )

    if chains_idx is None:
        chains_idx = []
        for k in range(len(chains_atom_pos)):
            chains_idx.append( k )

    struct = StructureBuilder()
    struct.init_structure("1")
    struct.init_seg("1")
    struct.init_model("1")

    n_total_atom = 0
    for k in range(len(chains_atom_pos)):
        # For each chain
        chain_atom_pos = chains_atom_pos[k]
        chain_atom_mask = chains_atom_mask[k]

        chain_res_type = chains_res_type[k]
        chain_res_idx = chains_res_idx[k]
        chain_idx = chains_idx[k]
        chain_bfactor = chains_bfactor[k]

        # Init a new chain
        struct.init_chain(chain_names[chain_idx])

        for i in range(len(chain_atom_pos)):
            # For each residue
            res_type = chain_res_type[i]
            res_idx = chain_res_idx[i]

            if 21 <= res_type <= 28:
                res_name_1 = cc.restypes[res_type].upper()
                # DA DC DG DU A C G U
                res_name_3 = res_name_1
            else:
                raise NotImplementedError

            atom_names = nc.restype_name_to_compact_atom_names[res_name_1]

            n_atom = len(chain_atom_pos[i])
            if len(atom_names) > n_atom:
                atom_names = atom_names[:n_atom]

            field_name = " "
            struct.init_residue(res_name_3, field_name, res_idx, " ")

            for atom_name, pos, bfactor, mask in zip(
                atom_names, chain_atom_pos[i], chain_bfactor[i], chain_atom_mask[i]
            ):
                #print(atom_name)
                #print(pos)
                #print(bfactor)
                #print(mask)

                if atom_name is None or \
                   atom_name == "" or \
                   mask < 1 or \
                   np.any(np.isnan( pos )) or \
                   np.all(np.abs(pos) < 1e-3):
                    continue

                struct.set_line_counter(n_total_atom + 1)

                struct.init_atom(
                    name=atom_name,
                    coord=pos,
                    b_factor=bfactor,
                    occupancy=1.0,
                    altloc=" ",
                    fullname=atom_name,
                    element=atom_name[0],
                )
                n_total_atom += 1

    struct = struct.get_structure()
    if suffix in ['cif', '.cif', 'CIF', '.CIF', 'mmcif', 'MMCIF']:
        io = CIFXIO()
        io.set_structure(struct)
        io.save(filename)

        # Fix quotes
        fix_quotes(filename)
    else:
        io = PDBIO()
        io.set_structure(struct)
        io.save(filename, write_end=False)


def chains_atom_pos_to_data(
    filename,
    chains_atom_pos,
    chains_atom_mask,
    chains_res_type,
    chains_res_idx=None,
    chains_idx=None, 
    chains_bfactor=None,
    chains_occupancy=None,
    remarks=None,
):
    # For different chains
    assert len(chains_atom_pos) == len(chains_atom_mask)
    if chains_occupancy is None:
        chains_occupancy = []
        for k in range(len(chains_atom_pos)):
            chains_occupancy.append( np.full(len(chains_atom_pos[k]), 1.0, dtype=np.float32) )

    if chains_bfactor is None:
        chains_bfactor = []
        for k in range(len(chains_atom_pos)):
            chains_bfactor.append( np.ones_like(chains_atom_mask[k], dtype=np.float32) * 100.0 )

    if chains_res_type is None:
        chains_res_type = []
        for k in range(len(chains_atom_pos)):
            chains_res_type.append( np.full(len(chains_atom_pos[k]), 0, dtype=np.int32) )

    if chains_res_idx is None:
        chains_res_idx = []
        for k in range(len(chains_atom_pos)):
            chains_res_idx.append( np.arange(len(chains_atom_pos[k]), dtype=np.int32) )

    if chains_idx is None:
        chains_idx = []
        for k in range(len(chains_atom_pos)):
            chains_idx.append( k )

    with open(filename, 'w') as f:
        n_total_atom = 0
        for k in range(len(chains_atom_pos)):
            # For each chain
            chain_atom_pos = chains_atom_pos[k]
            chain_atom_mask = chains_atom_mask[k]

            chain_res_type = chains_res_type[k]
            chain_res_idx = chains_res_idx[k]
            chain_idx = chains_idx[k]
            chain_bfactor = chains_bfactor[k]

            for i in range(len(chain_atom_pos)):
                # For each residue
                res_type = chain_res_type[i]
                res_idx = chain_res_idx[i]

                if 21 <= res_type <= 28:
                    res_name_1 = cc.restypes[res_type].upper()
                    # DA DC DG DU A C G U
                    res_name_3 = res_name_1
                else:
                    raise NotImplementedError

                atom_names = nc.restype_name_to_compact_atom_names[res_name_1]

                n_atom = len(chain_atom_pos[i])
                if len(atom_names) > n_atom:
                    atom_names = atom_names[:n_atom]

                for atom_name, pos, bfactor, mask in zip(
                    atom_names, chain_atom_pos[i], chain_bfactor[i], chain_atom_mask[i]
                ):

                    if atom_name is None or \
                       atom_name == "" or \
                       mask < 1 or \
                       np.any(np.isnan( pos )) or \
                       np.all(np.abs(pos) < 1e-3):
                        continue

                    # write atom
                    f.write("{:<7s} {:<4s} {:<8d} {:8.3f} {:8.3f} {:8.3f}\n".format(
                        "ATOM",
                        atom_name.strip(),
                        n_total_atom,
                        pos[0],
                        pos[1],
                        pos[2],
                    ))

                    n_total_atom += 1

                # write residue
                f.write("{:<7s} {:<4s} {:<8d}\n".format(
                    "RESIDUE",
                    res_name_3,
                    i, 
                ))

            # write chain
            f.write("{:<7s} {:<4s} {:<8d}\n".format(
                "CHAIN",
                chain_names[chain_idx],
                chain_idx, 
            ))

        # write model
        f.write("{:<7s} {:<4s} {:<8d}\n".format(
            "MODEL",
            "A",
            0, 
        ))


# read pdb
def read_pdb(
    filename, 
    ignore_hetatm=False,
):
    # Read file
    if filename.endswith("pdb"):
        parser = PDBParser()
    elif filename.endswith("cif"):
        parser = MMCIFParser()
    else:
        raise Exception("Error only support pdb/cif file")
    structure = parser.get_structure('pdb', filename)

    # Only use model 0
    model = structure[0]

    # Extract residue information
    atom_pos = []
    atom_mask = []
    res_type = []
    res_idx = []
    chain_idx = []
    bfactor = []

    for i, chain in enumerate(model):
        prev_residue_number = None
        for residue_index, residue in enumerate(chain):
            residue_number = residue.get_id()[1]
            if prev_residue_number is None or residue_number != prev_residue_number:
                resname_3 = residue.get_resname().strip()
                hetfield, resseq, icode = residue.get_id()

                # If hetatom
                if ignore_hetatm and hetfield != " ":
                    continue

                # If unknown residues
                if resname_3 not in cc.restype_3to1:
                    continue

                curr_res_type = resname_1 = cc.restype_1_to_index[cc.restype_3to1[resname_3]]

                # For NA and protein
                if 21 <= curr_res_type <= 28:
                    atom_names = nc.restype_name_to_compact_atom_names[resname_3]
                else:
                    atom_names = pc.restype_name_to_compact_atom_names[resname_3]
                    continue

                #print(resname_3, curr_res_type, atom_names)

                prev_residue_number = residue_number

            coords = []
            mask = []
            bf = []

            # Get atom types for current residue
            while len(atom_names) < 23:
                atom_names.append("")

            for atom_index in atom_names:
                try:
                    atom = residue[atom_index]
                    coords.append(atom.get_coord())
                    mask.append(1)
                    bf.append(atom.get_bfactor())
                except KeyError:
                    coords.append([float("nan") for _ in range(3)])
                    mask.append(0)
                    bf.append(float("nan"))

            atom_pos.append(coords)
            atom_mask.append(mask)
            chain_idx.append(i)
            res_type.append(curr_res_type)
            res_idx.append(residue_number)

            bfactor.append(bf)

    # Convert to NumPy arrays
    atom_pos = np.array(atom_pos).astype(np.float32)
    atom_mask = np.array(atom_mask).astype(np.int32)
    res_type = np.array(res_type).astype(np.int32)
    res_idx = np.array(res_idx).astype(np.int32)
    chain_idx = np.array(chain_idx).astype(np.int32)
    bfactor = np.array(bfactor).astype(np.float32)

    # only keep na targets
    mask = np.logical_and(res_type >= 21, res_type <= 28)
    #print(atom_mask.shape)
    #print(mask.shape)
    atom_pos = atom_pos[mask]
    atom_mask = atom_mask[mask]
    res_type = res_type[mask]
    res_idx = res_idx[mask]
    chain_idx = chain_idx[mask]
    bfactor = bfactor[mask]

    chain_idx = chain_idx - chain_idx.min()

    return atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor



def extend_torch(x: torch.Tensor, n: int):
    # x: (..., k, d)
    k = x.shape[-2]
    if k >= n:
        return x[..., :n, :]
    else:
        pad_shape = list(x.shape[:-2]) + [n - k, x.shape[-1]]
        pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-2)

def na_make_atom27_to_atom23_index_torch():
    atom27_to_atom23_idx = dict()

    for resname in nc.restypes:
        atoms = nc.restype_name_to_compact_atom_names[resname]

        res_atom27_to_atom23_idx = []
        for atom in atoms:
            try:
                idx = nc.atom_types.index(atom)
            except ValueError:
                idx = -1

            res_atom27_to_atom23_idx.append(idx)

        atom27_to_atom23_idx[resname] = res_atom27_to_atom23_idx

    return atom27_to_atom23_idx

def atom27_to_atom23_torch(atom_pos, atom_mask, res_type):
    """
    Args:
        atom_pos: (L, 27, 3) torch.Tensor
        atom_mask: (L, 27) torch.Tensor
        res_type: (L,) torch.LongTensor
    Returns:
        atom_pos_23: (L, 23, 3)
        atom_mask_23: (L, 23)
    """
    device = atom_pos.device
    L = atom_pos.shape[0]

    res_to_idx = na_make_atom27_to_atom23_index_torch()  # dict[str] -> List[int]
    
    idx_map_tensor = torch.full((L, 23), -1, dtype=torch.long, device=device)

    for i in range(L):
        res = res_type[i].item()
        resname_1 = cc.restype_index_to_1[res]
        if 21 <= res <= 28:
            resname_1 = resname_1.upper()
        else:
            raise NotImplementedError

        idxs = res_to_idx[resname_1]
        n = len(idxs)
        idx_tensor = torch.tensor(idxs, device=device, dtype=torch.long)
        idx_map_tensor[i, :n] = idx_tensor

    gather_index = idx_map_tensor.unsqueeze(-1).expand(-1, -1, 3)  # (L, 23, 3)
    atom_pos_23 = torch.gather(atom_pos, dim=1, index=torch.clamp(gather_index, 0))

    atom_mask_idx = idx_map_tensor.clone()
    atom_mask_idx[idx_map_tensor == -1] = 0
    atom_mask_23 = torch.gather(atom_mask, dim=1, index=atom_mask_idx)  # (L, 23)

    atom_mask_23 = atom_mask_23 * (idx_map_tensor != -1).float()
    atom_pos_23 = atom_pos_23 * atom_mask_23.unsqueeze(-1)

    return atom_pos_23, atom_mask_23



