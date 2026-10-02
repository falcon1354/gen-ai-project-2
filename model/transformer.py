"""
transformer.py
--------------
Full encoder-decoder Transformer.

Config (from manual):
  d_model = 256
  n_heads = 4
  d_k = d_v = 64
  n_encoder_layers = 3
  n_decoder_layers = 3
  d_ff = 1024
  dropout = 0.1
  post-norm
  weight sharing = Yes  (src emb = tgt emb = output projection^T idea)
"""

import math
import os
import sys

import torch
import torch.nn as nn

# allow both:  python -m model.transformer   and   from model.transformer import ...
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.layers import EncoderLayer, DecoderLayer

# reuse starter embeddings
from starter.embeddings import TokenEmbedding, PositionalEncoding


def make_padding_mask(seq, pad_id):
    """
    seq: (batch, seq_len)
    returns mask: (batch, 1, 1, seq_len)
      1 = real token (keep)
      0 = pad token (mask out)
    """
    # (batch, seq_len) -> True where NOT pad
    mask = (seq != pad_id).unsqueeze(1).unsqueeze(2)
    return mask.long()


def make_causal_mask(size, device):
    """
    Look-ahead / causal mask for decoder self-attention.
    size = tgt_len

    returns: (1, 1, size, size)
      lower triangle = 1 (can see)
      upper triangle = 0 (cannot see future)
    """
    # ones on and below diagonal
    mask = torch.tril(torch.ones(size, size, device=device)).long()
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


def make_tgt_mask(tgt, pad_id):
    """
    Decoder self-attention mask = padding mask AND causal mask.
    tgt: (batch, tgt_len)
    """
    tgt_len = tgt.size(1)
    device = tgt.device

    pad_mask = make_padding_mask(tgt, pad_id)          # (batch, 1, 1, tgt_len)
    causal = make_causal_mask(tgt_len, device)         # (1, 1, tgt_len, tgt_len)

    # broadcast multiply / AND
    # pad_mask needs to cover keys; for queries we also shouldn't attend from pads,
    # but key masking is the important part for loss ignore later.
    tgt_mask = pad_mask & causal
    return tgt_mask


class Encoder(nn.Module):
    """Stack of N encoder layers."""

    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        i = 0
        while i < n_layers:
            self.layers.append(EncoderLayer(d_model, n_heads, d_ff, dropout))
            i = i + 1
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        attn_list = []
        i = 0
        while i < len(self.layers):
            x, attn = self.layers[i](x, src_mask=src_mask)
            attn_list.append(attn)
            i = i + 1
        x = self.norm(x)
        return x, attn_list


class Decoder(nn.Module):
    """Stack of N decoder layers."""

    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        i = 0
        while i < n_layers:
            self.layers.append(DecoderLayer(d_model, n_heads, d_ff, dropout))
            i = i + 1
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        self_attn_list = []
        cross_attn_list = []
        i = 0
        while i < len(self.layers):
            x, self_attn, cross_attn = self.layers[i](
                x, enc_out, tgt_mask=tgt_mask, memory_mask=memory_mask
            )
            self_attn_list.append(self_attn)
            cross_attn_list.append(cross_attn)
            i = i + 1
        x = self.norm(x)
        return x, self_attn_list, cross_attn_list


class Transformer(nn.Module):
    """
    Full Text-to-SQL Transformer.

    forward(src_ids, tgt_ids):
      src_ids: (batch, src_len)   encoder tokens
      tgt_ids: (batch, tgt_len)   decoder input tokens  (already teacher-forced[:-1] OR growing during decode)

    returns:
      logits: (batch, tgt_len, vocab_size)
      also attention maps if needed
    """

    def __init__(
        self,
        vocab_size,
        pad_id,
        d_model=256,
        n_heads=4,
        n_encoder_layers=3,
        n_decoder_layers=3,
        d_ff=1024,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()

        self.pad_id = pad_id
        self.d_model = d_model
        self.vocab_size = vocab_size

        # shared token embedding for encoder + decoder
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        self.encoder = Encoder(n_encoder_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_decoder_layers, d_model, n_heads, d_ff, dropout)

        # output projection to vocab
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # ----- weight sharing -----
        # share embedding weights with output projection (classic Transformer trick)
        self.output_proj.weight = self.token_embedding.embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Simple Xavier init for linear layers."""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        # re-tie after init (init may have overwritten)
        self.output_proj.weight = self.token_embedding.embedding.weight

    def embed(self, token_ids):
        """Token embed + scale is inside TokenEmbedding; then add PE."""
        x = self.token_embedding(token_ids)
        x = self.pos_encoding(x)
        return x

    def forward(self, src_ids, tgt_ids, return_attn=False):
        src_mask = make_padding_mask(src_ids, self.pad_id)
        tgt_mask = make_tgt_mask(tgt_ids, self.pad_id)
        memory_mask = src_mask  # decoder cross-attn should not look at src pads

        src = self.embed(src_ids)
        tgt = self.embed(tgt_ids)

        enc_out, enc_attn = self.encoder(src, src_mask=src_mask)
        dec_out, self_attn, cross_attn = self.decoder(
            tgt, enc_out, tgt_mask=tgt_mask, memory_mask=memory_mask
        )

        logits = self.output_proj(dec_out)

        if return_attn:
            return logits, {
                "enc_attn": enc_attn,
                "dec_self_attn": self_attn,
                "dec_cross_attn": cross_attn,
            }
        return logits

    def encode(self, src_ids):
        """Encode source once (useful during decoding)."""
        src_mask = make_padding_mask(src_ids, self.pad_id)
        src = self.embed(src_ids)
        enc_out, _ = self.encoder(src, src_mask=src_mask)
        return enc_out, src_mask

    def decode_step(self, tgt_ids, enc_out, src_mask, return_attn=False):
        """
        Run decoder on current tgt prefix.
        tgt_ids: (batch, cur_len)
        """
        tgt_mask = make_tgt_mask(tgt_ids, self.pad_id)
        tgt = self.embed(tgt_ids)
        dec_out, self_attn, cross_attn = self.decoder(
            tgt, enc_out, tgt_mask=tgt_mask, memory_mask=src_mask
        )
        logits = self.output_proj(dec_out)
        if return_attn:
            return logits, cross_attn
        return logits


def count_parameters(model):
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total = total + p.numel()
    return total


def smoke_test():
    """Quick shape check with fake data."""
    vocab_size = 100
    pad_id = 0
    batch = 2
    src_len = 12
    tgt_len = 8

    model = Transformer(vocab_size=vocab_size, pad_id=pad_id)
    src = torch.randint(1, vocab_size, (batch, src_len))
    tgt = torch.randint(1, vocab_size, (batch, tgt_len))
    # put some pads
    src[:, -2:] = pad_id
    tgt[:, -1:] = pad_id

    logits = model(src, tgt)
    print("logits", tuple(logits.shape))  # (2, 8, 100)
    print("params", count_parameters(model))

    logits2, attns = model(src, tgt, return_attn=True)
    last_cross = attns["dec_cross_attn"][-1]
    print("last cross-attn", tuple(last_cross.shape))


if __name__ == "__main__":
    smoke_test()
