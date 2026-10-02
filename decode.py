"""
decode.py
---------
Greedy decoding and beam search (beam size = 4).

Stop when </s> is generated or after max_len tokens (default 64).
"""

import os
import sys

import torch
import torch.nn.functional as F
import sentencepiece as spm

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.transformer import Transformer
from sql_parse import parse_generated_sql, sql_to_readable


def load_tokenizer(sp_model_path):
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)
    return sp


def load_model(checkpoint_path, device):
    """Load Transformer from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = Transformer(
        vocab_size=ckpt["vocab_size"],
        pad_id=ckpt["pad_id"],
        d_model=ckpt.get("d_model", 256),
        n_heads=ckpt.get("n_heads", 4),
        n_encoder_layers=ckpt.get("n_encoder_layers", 3),
        n_decoder_layers=ckpt.get("n_decoder_layers", 3),
        d_ff=ckpt.get("d_ff", 1024),
        dropout=ckpt.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt


def ids_to_text(sp, ids, bos_id, eos_id, pad_id):
    """Convert id list to string, removing special tokens."""
    clean = []
    i = 0
    while i < len(ids):
        tok = ids[i]
        if tok == bos_id or tok == eos_id or tok == pad_id:
            i = i + 1
            continue
        clean.append(tok)
        i = i + 1
    return sp.DecodeIds(clean)


@torch.no_grad()
def greedy_decode(model, src_ids, bos_id, eos_id, max_len=64):
    """
    src_ids: (1, src_len) tensor
    returns: list of token ids including bos...eos (eos may be missing if max_len hit)
    """
    device = src_ids.device
    enc_out, src_mask = model.encode(src_ids)

    # start with <s>
    ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    t = 0
    while t < max_len:
        logits = model.decode_step(ys, enc_out, src_mask)
        # logits: (1, cur_len, vocab)
        next_logits = logits[:, -1, :]
        next_id = torch.argmax(next_logits, dim=-1)  # (1,)

        ys = torch.cat([ys, next_id.unsqueeze(0)], dim=1)

        if int(next_id.item()) == eos_id:
            break
        t = t + 1

    return ys[0].tolist()


@torch.no_grad()
def beam_search_decode(model, src_ids, bos_id, eos_id, pad_id, beam_size=4, max_len=64):
    """
    Simple beam search.

    Keeps `beam_size` best partial sequences by sum of log-probs.
    """
    device = src_ids.device
    enc_out, src_mask = model.encode(src_ids)

    # each beam: (token_ids_list, log_prob_sum, finished)
    beams = [([bos_id], 0.0, False)]

    step = 0
    while step < max_len:
        # if all finished, stop
        all_done = True
        bi = 0
        while bi < len(beams):
            if beams[bi][2] is False:
                all_done = False
            bi = bi + 1
        if all_done:
            break

        candidates = []

        bi = 0
        while bi < len(beams):
            tokens, score, done = beams[bi]

            if done:
                candidates.append((tokens, score, True))
                bi = bi + 1
                continue

            ys = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model.decode_step(ys, enc_out, src_mask)
            next_logits = logits[:, -1, :]  # (1, vocab)
            log_probs = F.log_softmax(next_logits, dim=-1)[0]  # (vocab,)

            # take top beam_size tokens for this beam
            top_log_probs, top_ids = torch.topk(log_probs, k=beam_size)

            k = 0
            while k < beam_size:
                tok = int(top_ids[k].item())
                new_score = score + float(top_log_probs[k].item())
                new_tokens = list(tokens)
                new_tokens.append(tok)
                finished = (tok == eos_id)
                candidates.append((new_tokens, new_score, finished))
                k = k + 1

            bi = bi + 1

        # sort candidates by score (higher is better)
        def score_of(item):
            return item[1]

        candidates.sort(key=score_of, reverse=True)

        # keep best beam_size
        beams = candidates[:beam_size]
        step = step + 1

    # pick best finished beam if any, else best overall
    best = beams[0]
    i = 0
    while i < len(beams):
        tokens, score, done = beams[i]
        if done:
            if (best[2] is False) or (score > best[1]):
                best = beams[i]
        i = i + 1

    return best[0]


def decode_one(
    model,
    sp,
    source_text,
    device,
    method="beam",
    beam_size=4,
    max_len=64,
    headers=None,
):
    """
    High-level helper:
      source_text -> generated SQL text (+ optional readable SQL)
    """
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    pad_id = sp.pad_id()

    src_ids_list = sp.EncodeAsIds(source_text)
    src = torch.tensor([src_ids_list], dtype=torch.long, device=device)

    if method == "greedy":
        out_ids = greedy_decode(model, src, bos_id, eos_id, max_len=max_len)
    else:
        out_ids = beam_search_decode(
            model, src, bos_id, eos_id, pad_id, beam_size=beam_size, max_len=max_len
        )

    gen_text = ids_to_text(sp, out_ids, bos_id, eos_id, pad_id)
    sql_dict, err = parse_generated_sql(gen_text)

    readable = None
    if sql_dict is not None and headers is not None:
        readable = sql_to_readable(sql_dict, headers)

    return {
        "generated_text": gen_text,
        "sql_dict": sql_dict,
        "parse_error": err,
        "readable_sql": readable,
        "token_ids": out_ids,
    }


if __name__ == "__main__":
    print("decode.py helpers ready. Use train.py first to create a checkpoint.")
