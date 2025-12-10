import os
import argparse
import subprocess
import numpy as np
import torch

from em3na.utils.torch_utils import get_device_names
from em3na.io.seqio import read_fasta

def get_pretrained_model(
    model_dir: str, 
    model_name: str, 
):
    from rinalmo.data.alphabet import Alphabet
    from rinalmo.model.model import RiNALMo
    from rinalmo.config import model_config

    if "giga" in model_name:
        lm_config = "giga"
    elif "mega" in model_name:
        lm_config = "mega"
    elif "micro" in model_name:
        lm_config = "micro"
    else:
        raise RuntimeError("Wrong model name")

    pretrained_weights_path = os.path.join(model_dir, model_name)

    config = model_config(lm_config)
    model = RiNALMo(config)
    alphabet = Alphabet(**config['alphabet'])
    model.load_state_dict(torch.load(pretrained_weights_path))

    return model, alphabet

def add_args(parser):
    script_dir = os.path.abspath(os.path.dirname(__file__))
    parser.add_argument("--rna", "-r", type=str)
    parser.add_argument("--dna", "-d", type=str)
    #parser.add_argument("--seq", "-s", type=str, required=True)

    parser.add_argument("--output-dir", "-o", type=str, default=".")
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument(
        "--model-dir", 
        type=str, 
        default=os.path.join(script_dir, "..", "rinalmo", "weights"),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="rinalmo_giga_pretrained.pt",
    )
    parser.add_argument("--device", default="0", help="Running device")
    return parser

def main(args):
    # read sequence
    rna_seqs = []
    if os.path.exists(args.rna):
        rna_seqs = read_fasta(args.rna)
        print("# Done read RNA seqs")

    dna_seqs = []
    if os.path.exists(args.dna):
        dna_seqs = read_fasta(args.dna)
        print("# Done read DNA seqs")

    # simple concat
    seqs = rna_seqs + dna_seqs

    assert len(seqs) > 0, "Cannot read any seqs from fasta file"
    new_seq = []
    for seq in seqs:
        for s in seq:
            if s == "T":
                s = "U"
            if s not in ["A", "C", "G", "U"]:
                s = "N"
            new_seq.append(s)
    seq = "".join(new_seq)
    print("# Total seq length = {}".format(len(seq)))

    # split too long seqs into sub-seqs
    sub_seqs = []
    for i in range(0, len(seq), args.max_length):
        sub_seqs.append( seq[i : i + args.max_length] )
    print("# Split into {} sub-seqs".format(len(sub_seqs)))

    # get model
    devices = get_device_names(args.device)
    model, alphabet = get_pretrained_model(
        model_dir=args.model_dir, 
        model_name=args.model_name,
    )
    model = model.to(device=devices[0])
    model.eval()
    print("# Done read model")

    # inference
    print("# Run inference")
    repr_list = []
    for seq in sub_seqs:
        seqs = [seq]
        tokens = torch.tensor(alphabet.batch_tokenize(seqs), dtype=torch.int64, device=devices[0])
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model(tokens, need_attn_weights=False)
        repr_list.append( outputs['representation'][0][1:-1] )
    representation = torch.cat(repr_list, dim=0).float().cpu().numpy()

    # output
    os.makedirs(args.output_dir, exist_ok=True)
    fo = os.path.join(args.output_dir, "seq_embed.npy")
    np.save(fo, representation)
    print("# Save seq embed to {}".format(fo))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)

