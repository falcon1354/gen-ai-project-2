"""
check_model.py
--------------
Quick shape / parameter check for the Transformer.
Run after starter check passes.
"""

import os
import sys

import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from starter.dataset import make_dataloader, get_default_paths
from model.transformer import Transformer, count_parameters


def main():
    train_pairs, _, sp_model = get_default_paths()
    if not os.path.exists(train_pairs) or not os.path.exists(sp_model):
        print("Run data_prep.py and tokenizer.py first.")
        return

    loader, dataset = make_dataloader(train_pairs, sp_model, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    src = batch["src"]
    tgt = batch["tgt"]
    decoder_in = tgt[:, :-1]
    labels = tgt[:, 1:]

    model = Transformer(vocab_size=dataset.sp.GetPieceSize(), pad_id=dataset.pad_id)
    logits, attns = model(src, decoder_in, return_attn=True)

    print("src         ", tuple(src.shape))
    print("decoder_in  ", tuple(decoder_in.shape))
    print("labels      ", tuple(labels.shape))
    print("logits      ", tuple(logits.shape))
    print("params      ", count_parameters(model))
    print("last cross  ", tuple(attns["dec_cross_attn"][-1].shape))
    print("OK")


if __name__ == "__main__":
    main()
