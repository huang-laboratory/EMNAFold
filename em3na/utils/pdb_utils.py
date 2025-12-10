import numpy as np

def load_translation_from_file(filename, atom="C4'"):
    with open(filename, 'r') as f:
        lines = f.readlines()

    trans = []
    if filename.endswith(".pdb"):
        for line in lines:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            assert len(line) >= 54, "Wrong PDB format"
            atom_name = line[12:16].strip()
            if atom is not None and atom_name != atom:
                continue

            trans.append(np.asarray([float(x) for x in [line[i: i + 8] for i in [30, 38, 46]]], dtype=np.float32))

    elif filename.endswith(".cif"):
        for line in lines:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            fields = line.strip().split()
            assert len(fields) >= 13, "Wrong CIF format"
            atom_name = fields[3]
            if atom is not None and (atom not in atom_name):
                continue

            trans.append(np.array([fields[10], fields[11], fields[12]], dtype=np.float32))
            
    else:
        raise NotImplementedError
    trans = np.asarray(trans, dtype=np.float32)

    return trans
