"""
embeddings.py
-------------
Token embeddings + sinusoidal positional encoding + input layer.

This is the last provided starter piece. After this, your Transformer
receives tensors shaped:
  encoder input: (batch, src_len, d_model)
  decoder input: (batch, tgt_len - 1, d_model)
"""

import math
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class TokenEmbedding(nn.Module):
    """Simple lookup table from token id -> d_model vector."""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids):
        # token_ids: (batch, seq_len)
        # out:       (batch, seq_len, d_model)
        out = self.embedding(token_ids)

        # scale by sqrt(d_model) like the original Transformer paper
        out = out * math.sqrt(self.d_model)
        return out


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encodings.

    PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i + 1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # make a big table of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1).float()
        # position shape: (max_len, 1)

        # div_term for even indices
        # 10000^(2i/d_model)  -->  exp( (2i) * (-log(10000)/d_model) )
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        # even dimensions: sin
        pe[:, 0::2] = torch.sin(position * div_term)

        # odd dimensions: cos
        # if d_model is odd this still works because of slicing
        pe[:, 1::2] = torch.cos(position * div_term)

        # add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # register as buffer so it moves with .to(device) but is not a parameter
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)

        # add positional encoding
        x = x + self.pe[:, :seq_len, :]
        x = self.dropout(x)
        return x

    def get_pe_matrix(self, length=100):
        """Return pe table for plotting: (length, d_model)"""
        return self.pe[0, :length, :].detach().cpu()


class InputLayer(nn.Module):
    """
    Combines token embedding + positional encoding.

    Used for BOTH encoder and decoder inputs.
    Weight sharing later can reuse the same TokenEmbedding.
    """

    def __init__(self, vocab_size, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

    def forward(self, token_ids):
        # token_ids: (batch, seq_len)
        x = self.token_embedding(token_ids)
        x = self.pos_encoding(x)
        return x


def save_positional_encoding_heatmap(pos_encoding, out_path, length=50, dims=64):
    """
    Save a heatmap of the positional encoding matrix.
    Manual asks for this plot.
    """
    pe = pos_encoding.get_pe_matrix(length=length)
    # only show first `dims` dimensions so the plot is readable
    pe_show = pe[:, :dims].numpy()

    plt.figure(figsize=(10, 6))
    plt.imshow(pe_show, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.xlabel("Embedding dimension")
    plt.ylabel("Position")
    plt.title("Positional Encoding Heatmap")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved positional encoding heatmap to:", out_path)


if __name__ == "__main__":
    # tiny shape check without real data
    vocab_size = 100
    d_model = 256
    batch = 4
    src_len = 10
    tgt_len = 8

    layer = InputLayer(vocab_size, d_model)
    src_ids = torch.randint(0, vocab_size, (batch, src_len))
    tgt_ids = torch.randint(0, vocab_size, (batch, tgt_len))

    enc_in = layer(src_ids)
    dec_in = layer(tgt_ids)

    print("encoder input", tuple(enc_in.shape))
    print("decoder input", tuple(dec_in.shape))

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    out_path = os.path.join(results_dir, "positional_encoding_heatmap.png")
    save_positional_encoding_heatmap(layer.pos_encoding, out_path)
