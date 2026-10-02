"""
attention.py
------------
Scaled dot-product attention and multi-head attention.

Beginner style: clear shapes in comments, no fancy tricks.

Shapes reminder (one head):
  Q: (batch, heads, q_len, d_k)
  K: (batch, heads, k_len, d_k)
  V: (batch, heads, k_len, d_v)

  scores = Q @ K^T / sqrt(d_k)   -> (batch, heads, q_len, k_len)
  attn   = softmax(scores)
  out    = attn @ V              -> (batch, heads, q_len, d_v)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """One attention operation: score, mask, softmax, weighted sum of V."""

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        q: (batch, heads, q_len, d_k)
        k: (batch, heads, k_len, d_k)
        v: (batch, heads, k_len, d_v)
        mask: broadcastable to (batch, heads, q_len, k_len)
              True / 1 means KEEP, False / 0 means MASK OUT
              (we will treat mask == 0 as blocked)
        """
        d_k = q.size(-1)

        # (batch, heads, q_len, k_len)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(d_k)

        if mask is not None:
            # replace blocked positions with a very negative number
            # so softmax becomes ~0 there
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # (batch, heads, q_len, d_v)
        out = torch.matmul(attn, v)
        return out, attn


class MultiHeadAttention(nn.Module):
    """
    Split d_model into H heads, run attention, then merge again.

    For this project:
      d_model = 256
      n_heads = 4
      d_k = d_v = 64
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        # d_model must split evenly across heads
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        # one big linear for Q, K, V (easier than 3 separate for beginners too)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

        # final projection after concatenating heads
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        x: (batch, seq_len, d_model)
        return: (batch, n_heads, seq_len, d_k)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # (batch, seq_len, n_heads, d_k)
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)

        # (batch, n_heads, seq_len, d_k)
        x = x.transpose(1, 2)
        return x

    def merge_heads(self, x):
        """
        x: (batch, n_heads, seq_len, d_k)
        return: (batch, seq_len, d_model)
        """
        batch_size = x.size(0)
        seq_len = x.size(2)

        # (batch, seq_len, n_heads, d_k)
        x = x.transpose(1, 2).contiguous()

        # (batch, seq_len, d_model)
        x = x.view(batch_size, seq_len, self.d_model)
        return x

    def forward(self, query, key, value, mask=None):
        """
        query/key/value: (batch, seq_len, d_model)

        For self-attention: query=key=value=same tensor
        For cross-attention: query=decoder, key=value=encoder output
        """
        batch_size = query.size(0)

        # project
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # split into heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # mask shape notes:
        # padding mask often comes as (batch, 1, 1, src_len)
        # causal mask as (1, 1, tgt_len, tgt_len)
        # both broadcast over heads
        out, attn = self.attention(q, k, v, mask=mask)

        # merge heads and project
        out = self.merge_heads(out)
        out = self.w_o(out)
        # dropout is applied in EncoderLayer / DecoderLayer around residual
        return out, attn
