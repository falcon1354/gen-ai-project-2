"""
layers.py
---------
Feed-forward network, EncoderLayer, DecoderLayer.

Manual requires POST-NORM:
  x = LayerNorm(x + sublayer(x))
"""

import os
import sys

import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise FFN:
      Linear(d_model -> d_ff) -> ReLU -> Dropout -> Linear(d_ff -> d_model)
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """
    One encoder block:

      self-attention
          ↓
      residual + LayerNorm
          ↓
      feed-forward
          ↓
      residual + LayerNorm
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # --- self attention (post-norm) ---
        attn_out, attn_weights = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # --- feed forward (post-norm) ---
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x, attn_weights


class DecoderLayer(nn.Module):
    """
    One decoder block (order from the manual):

      masked self-attention
          ↓
      cross-attention over encoder output
          ↓
      feed-forward

    Each step has residual + LayerNorm (post-norm).
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        # --- 1) masked self-attention ---
        self_out, self_attn = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_out))

        # --- 2) cross-attention (Q=decoder, K/V=encoder) ---
        cross_out, cross_attn = self.cross_attn(
            x, enc_out, enc_out, mask=memory_mask
        )
        x = self.norm2(x + self.dropout(cross_out))

        # --- 3) feed-forward ---
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        # return cross_attn for visualization (last layer usually)
        return x, self_attn, cross_attn
