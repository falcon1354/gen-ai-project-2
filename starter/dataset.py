"""
dataset.py
----------
Load processed src/tgt pairs, tokenize them, and make batches.

Teacher forcing reminder:
  full target ids:   [<s>, t1, t2, ..., tn, </s>]
  decoder input:     [<s>, t1, t2, ..., tn]        = tgt[:, :-1]
  prediction labels: [t1, t2, ..., tn, </s>]       = tgt[:, 1:]
"""

import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm


class TextToSQLDataset(Dataset):
    """Simple dataset: each item is one (src_ids, tgt_ids) pair."""

    def __init__(self, pairs_path, sp_model_path):
        self.examples = []

        # load SentencePiece
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(sp_model_path)

        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

        # read all pairs into memory (WikiSQL is small enough)
        with open(pairs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                row = json.loads(line)
                self.examples.append(row)

    def __len__(self):
        return len(self.examples)

    def encode_source(self, text):
        """Encode source text to id list. No BOS/EOS needed for encoder usually."""
        ids = self.sp.EncodeAsIds(text)
        return ids

    def encode_target(self, text):
        """Encode target with <s> at start and </s> at end."""
        ids = self.sp.EncodeAsIds(text)

        full = []
        full.append(self.bos_id)

        i = 0
        while i < len(ids):
            full.append(ids[i])
            i = i + 1

        full.append(self.eos_id)
        return full

    def __getitem__(self, index):
        row = self.examples[index]
        src_ids = self.encode_source(row["src"])
        tgt_ids = self.encode_target(row["tgt"])

        # return plain python lists; collate_fn will pad them
        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "table_id": row["table_id"],
        }


def pad_1d(sequences, pad_id):
    """
    Pad a list of 1D id lists to the same length.
    Returns a LongTensor of shape (batch, max_len)
    """
    batch_size = len(sequences)

    # find max length in this batch
    max_len = 0
    i = 0
    while i < batch_size:
        length = len(sequences[i])
        if length > max_len:
            max_len = length
        i = i + 1

    # make a tensor filled with pad_id
    out = torch.full((batch_size, max_len), pad_id, dtype=torch.long)

    i = 0
    while i < batch_size:
        seq = sequences[i]
        length = len(seq)
        j = 0
        while j < length:
            out[i, j] = seq[j]
            j = j + 1
        i = i + 1

    return out


def collate_fn(batch, pad_id):
    """
    Turn a list of dataset items into a batch of padded tensors.
    """
    src_list = []
    tgt_list = []
    table_ids = []

    i = 0
    while i < len(batch):
        src_list.append(batch[i]["src_ids"])
        tgt_list.append(batch[i]["tgt_ids"])
        table_ids.append(batch[i]["table_id"])
        i = i + 1

    src = pad_1d(src_list, pad_id)
    tgt = pad_1d(tgt_list, pad_id)

    return {
        "src": src,
        "tgt": tgt,
        "table_ids": table_ids,
    }


def make_dataloader(pairs_path, sp_model_path, batch_size=64, shuffle=True):
    """Create a DataLoader with our padding collate function."""
    dataset = TextToSQLDataset(pairs_path, sp_model_path)
    pad_id = dataset.pad_id

    def _collate(batch):
        return collate_fn(batch, pad_id)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate,
    )
    return loader, dataset


def get_default_paths():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(project_root, "starter", "processed")
    tok_dir = os.path.join(project_root, "starter", "tokenizer_files")

    train_pairs = os.path.join(processed_dir, "train_pairs.jsonl")
    dev_pairs = os.path.join(processed_dir, "dev_pairs.jsonl")
    sp_model = os.path.join(tok_dir, "spm.model")

    return train_pairs, dev_pairs, sp_model


if __name__ == "__main__":
    # tiny self-check
    train_pairs, dev_pairs, sp_model = get_default_paths()

    if not os.path.exists(train_pairs):
        print("Run data_prep.py first.")
    elif not os.path.exists(sp_model):
        print("Run tokenizer.py first.")
    else:
        loader, dataset = make_dataloader(train_pairs, sp_model, batch_size=64, shuffle=True)
        batch = next(iter(loader))
        print("src", tuple(batch["src"].shape))
        print("tgt", tuple(batch["tgt"].shape))
        print("num train examples:", len(dataset))
