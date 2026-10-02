"""
check_starter.py
----------------
Run this AFTER:
  1) WikiSQL data is in place
  2) data_prep.py
  3) tokenizer.py

It should print something close to the manual:

  train_pairs.jsonl: kept 56336, skipped 19
  ...
  src (64, ...)
  tgt (64, ...)
  encoder input (64, ..., 256)
  decoder input (64, ..., 256)
"""

import os
import sys

import torch

# make sure we can import sibling modules when run as a script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from dataset import make_dataloader, get_default_paths
from embeddings import InputLayer, save_positional_encoding_heatmap


def count_lines(path):
    """Count non-empty lines in a file."""
    if not os.path.exists(path):
        return 0

    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() != "":
                n = n + 1
    return n


def main():
    project_root = os.path.dirname(THIS_DIR)
    processed_dir = os.path.join(THIS_DIR, "processed")
    tok_dir = os.path.join(THIS_DIR, "tokenizer_files")
    results_dir = os.path.join(project_root, "results")

    train_pairs, dev_pairs, sp_model = get_default_paths()

    print("=== Checking files ===")
    print("train pairs :", train_pairs, "exists =", os.path.exists(train_pairs))
    print("dev pairs   :", dev_pairs, "exists =", os.path.exists(dev_pairs))
    print("spm model   :", sp_model, "exists =", os.path.exists(sp_model))

    if not os.path.exists(train_pairs) or not os.path.exists(dev_pairs):
        print("\nMissing pairs. Run: python starter/data_prep.py")
        return

    if not os.path.exists(sp_model):
        print("\nMissing tokenizer. Run: python starter/tokenizer.py")
        return

    print("\n=== Pair counts ===")
    print("train_pairs.jsonl lines:", count_lines(train_pairs))
    print("dev_pairs.jsonl lines  :", count_lines(dev_pairs))

    print("\n=== One training batch ===")
    loader, dataset = make_dataloader(
        train_pairs,
        sp_model,
        batch_size=64,
        shuffle=True,
    )

    batch = next(iter(loader))
    src = batch["src"]
    tgt = batch["tgt"]

    print("src", tuple(src.shape))
    print("tgt", tuple(tgt.shape))

    # teacher forcing: decoder sees all but last token
    decoder_token_ids = tgt[:, :-1]

    vocab_size = dataset.sp.GetPieceSize()
    d_model = 256

    input_layer = InputLayer(vocab_size=vocab_size, d_model=d_model, dropout=0.1)

    encoder_input = input_layer(src)
    decoder_input = input_layer(decoder_token_ids)

    print("encoder input", tuple(encoder_input.shape))
    print("decoder input", tuple(decoder_input.shape))

    # save positional encoding heatmap
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    heatmap_path = os.path.join(results_dir, "positional_encoding_heatmap.png")
    save_positional_encoding_heatmap(input_layer.pos_encoding, heatmap_path)

    print("\n=== Starter check finished OK ===")
    print("Next step: implement model/attention.py (scaled dot-product + multi-head).")


if __name__ == "__main__":
    main()
