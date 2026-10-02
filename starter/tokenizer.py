"""
tokenizer.py
------------
Train a SentencePiece tokenizer on our src/tgt text pairs.

We also add special tokens:
  <pad>  padding
  <s>    start of sequence
  </s>   end of sequence
  <unk>  unknown word
  <c0> ... <cN>  column placeholders

Run this AFTER data_prep.py
"""

import os
import json
import sentencepiece as spm


# how many column tokens to reserve
# WikiSQL tables are not huge, 50 is plenty
MAX_COLUMNS = 50

VOCAB_SIZE = 8000


def get_project_paths():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(project_root, "starter", "processed")
    tok_dir = os.path.join(project_root, "starter", "tokenizer_files")
    return project_root, processed_dir, tok_dir


def make_column_tokens():
    """Return list like ['<c0>', '<c1>', ...]."""
    tokens = []
    i = 0
    while i < MAX_COLUMNS:
        tokens.append("<c" + str(i) + ">")
        i = i + 1
    return tokens


def build_training_text(processed_dir, out_txt_path):
    """
    Put all train src + tgt text into one big text file.
    SentencePiece trains on this file.
    """
    train_pairs = os.path.join(processed_dir, "train_pairs.jsonl")

    if not os.path.exists(train_pairs):
        print("ERROR: train_pairs.jsonl not found.")
        print("Run data_prep.py first.")
        return False

    out = open(out_txt_path, "w", encoding="utf-8")

    with open(train_pairs, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            row = json.loads(line)
            out.write(row["src"] + "\n")
            out.write(row["tgt"] + "\n")

    out.close()
    print("Wrote corpus text:", out_txt_path)
    return True


def train_sentencepiece(corpus_path, model_prefix):
    """
    Train the SentencePiece model.

    We tell SentencePiece about our special tokens so it keeps them
    as whole pieces instead of splitting them.
    """
    column_tokens = make_column_tokens()

    # user_defined_symbols = tokens we always want as one piece
    # join with commas for the SentencePiece API
    user_symbols = []
    for t in column_tokens:
        user_symbols.append(t)

    # also keep SQL-ish words as whole tokens when possible
    extra_words = ["select", "where", "and", "max", "min", "count", "sum", "avg"]
    for w in extra_words:
        user_symbols.append(w)

    # Use keyword arguments (not one long string).
    # This avoids bugs when the folder path has spaces, e.g. "Gen AI Project 2".
    print("Training SentencePiece...")
    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        model_type="unigram",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        user_defined_symbols=user_symbols,
        # Do not add an extra space token at the start of every sentence
        add_dummy_prefix=False,
    )
    print("Saved model:", model_prefix + ".model")
    print("Saved vocab:", model_prefix + ".vocab")


def smoke_test(model_path):
    """Quick check that encoding/decoding works."""
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)

    sample = "select <c2> where <c0> = terrence ross"
    ids = sp.EncodeAsIds(sample)
    pieces = sp.EncodeAsPieces(sample)
    text_back = sp.DecodeIds(ids)

    # SentencePiece marks spaces with a special character.
    # Replace it so Windows terminals do not crash on print.
    pieces_nice = []
    for p in pieces:
        pieces_nice.append(p.replace("\u2581", "_"))

    print("Smoke test input :", sample)
    print("Pieces           :", pieces_nice)
    print("Ids              :", ids)
    print("Decoded back     :", text_back)
    print("Vocab size       :", sp.GetPieceSize())
    print("pad id           :", sp.pad_id())
    print("unk id           :", sp.unk_id())
    print("bos id           :", sp.bos_id())
    print("eos id           :", sp.eos_id())


def main():
    project_root, processed_dir, tok_dir = get_project_paths()

    if not os.path.exists(tok_dir):
        os.makedirs(tok_dir)

    corpus_path = os.path.join(tok_dir, "corpus.txt")
    model_prefix = os.path.join(tok_dir, "spm")

    ok = build_training_text(processed_dir, corpus_path)
    if not ok:
        return

    train_sentencepiece(corpus_path, model_prefix)
    smoke_test(model_prefix + ".model")
    print("Done.")


if __name__ == "__main__":
    main()
