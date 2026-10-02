"""
Streamlit demo for the Text-to-SQL Transformer.

Run:
  streamlit run app/streamlit_app.py
"""

import os
import sys

import streamlit as st
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from starter.data_prep import make_source, clean_text
from starter.dataset import get_default_paths
from decode import load_model, load_tokenizer, decode_one


@st.cache_resource
def load_resources(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, sp_model = get_default_paths()
    sp = load_tokenizer(sp_model)
    model, ckpt = load_model(checkpoint_path, device)
    return model, sp, device, ckpt


def main():
    st.set_page_config(page_title="Text-to-SQL Transformer", layout="centered")
    st.title("Text-to-SQL Transformer")
    st.caption("Encoder–decoder Transformer trained from scratch on WikiSQL")

    default_ckpt = os.path.join(_ROOT, "results", "best_model.pt")
    ckpt_path = st.text_input("Checkpoint path", value=default_ckpt)

    if not os.path.exists(ckpt_path):
        st.warning("Checkpoint not found yet. Train the model first (`python train.py`).")
        st.stop()

    model, sp, device, ckpt = load_resources(ckpt_path)

    question = st.text_input(
        "Question",
        value="What is Terrence Ross nationality?",
    )
    columns_raw = st.text_input(
        "Columns (comma-separated)",
        value="Player, No., Nationality, Position",
    )

    method = st.selectbox("Decoding", ["beam", "greedy"])

    if st.button("Generate SQL"):
        # build headers list
        parts = columns_raw.split(",")
        headers = []
        i = 0
        while i < len(parts):
            h = clean_text(parts[i])
            if h != "":
                headers.append(h)
            i = i + 1

        if question.strip() == "" or len(headers) == 0:
            st.error("Please enter a question and at least one column.")
            return

        src = make_source(question, headers)
        st.write("Model source text:")
        st.code(src)

        out = decode_one(
            model,
            sp,
            src,
            device,
            method=method,
            beam_size=4,
            headers=headers,
        )

        st.subheader("Generated text")
        st.code(out["generated_text"])

        st.subheader("Parsed WikiSQL")
        if out["sql_dict"] is None:
            st.error("Parse failed: " + str(out["parse_error"]))
        else:
            st.json(out["sql_dict"])

        st.subheader("Readable SQL")
        if out["readable_sql"] is None:
            st.write("(unavailable)")
        else:
            st.code(out["readable_sql"], language="sql")


if __name__ == "__main__":
    main()
