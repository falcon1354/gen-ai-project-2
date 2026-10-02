# Text-to-SQL Transformer (from scratch)

Build an **encoder–decoder Transformer in PyTorch**, train it on WikiSQL, evaluate it, then demo it in a small Streamlit app.

> Read every file before the viva. Submission marks come from understanding, not from having code.

## Folder layout

```text
your-project/
├── WikiSQL/
│   ├── data/                # jsonl + sqlite .db files
│   ├── lib/                 # official Query + DBEngine
│   └── evaluate.py          # official lf/ex accuracy script
├── starter/                 # data prep, tokenizer, dataset, embeddings
├── model/
│   ├── attention.py         # scaled dot-product + multi-head
│   ├── layers.py            # FFN, EncoderLayer, DecoderLayer
│   └── transformer.py       # masks + full model
├── sql_parse.py             # text -> WikiSQL dict -> readable SQL
├── train.py
├── decode.py                # greedy + beam search
├── evaluate_model.py
├── app/streamlit_app.py
├── results/                 # checkpoints + plots + metrics
├── download_wikisql.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python download_wikisql.py
python starter/data_prep.py
python starter/tokenizer.py
python starter/check_starter.py
```

## Model config (fixed by manual)

| Setting | Value |
|--------|------:|
| d_model | 256 |
| heads | 4 |
| d_k = d_v | 64 |
| encoder layers | 3 |
| decoder layers | 3 |
| FFN | 1024 |
| dropout | 0.1 |
| norm | post-norm |
| weight sharing | yes |

## Train (use GPU: Colab / Kaggle)

**Recommended:** open `colab_drive_train.ipynb` in Google Colab so checkpoints live on Drive and auto-resume after disconnect.

```bash
# saves to Drive folder; resumes automatically if last_model.pt exists
python train.py --epochs 20 --batch_size 64 --out_dir "/content/drive/MyDrive/GenAI_Project2/Gen AI Project 2/results" --resume
```

Saves:
- `last_model.pt`  (every epoch — used to resume)
- `best_model.pt`  (lowest dev loss — used for eval/app)

Force restart from scratch:

```bash
python train.py --epochs 20 --batch_size 64 --out_dir ".../results" --no_resume
```

Quick CPU smoke train:

```bash
python train.py --epochs 1 --max_train_batches 5 --batch_size 16 --no_resume
```

## Build Section 4 results tables/figures

```bash
python make_results_report.py
```

This writes `results/RESULTS.md` plus Table 1–4 JSON files and LR/loss plots.

To also fill missing greedy/test official scores (slow on CPU):

```bash
python make_results_report.py --run_missing_evals
```

Then:
1. Edit `results/training_meta.json` with your Colab GPU name + training hours  
2. Save a Streamlit screenshot as `results/frontend_screenshot.png`  
3. If you still need a real loss curve, either re-train with the updated `train.py` (logs `training_history.json`) or paste epoch losses into that file and re-run the report script

We use **Salesforce WikiSQL’s official** `WikiSQL/evaluate.py` for:

- `lf_accuracy` (logical form)
- `ex_accuracy` (execution)

```bash
python evaluate_model.py --split dev --method beam --beam_size 4 --make_attn_plot
python evaluate_model.py --split test --method beam --beam_size 4
```

This script:
1. writes `results/preds_dev.jsonl` in the official prediction format  
2. runs `WikiSQL/evaluate.py`  
3. also prints parse / select / agg / where breakdowns  

You can also run the official tool yourself (from `WikiSQL/`):

```bash
cd WikiSQL
python evaluate.py data/dev.jsonl data/dev.db ../results/preds_dev.jsonl
```

Metrics: `results/metrics_*.json`  
Attention plot: `results/cross_attention_heatmap.png`

## Frontend

```bash
streamlit run app/streamlit_app.py
```

## How data flows (know this for viva)

```text
question + column names
        ↓
SentencePiece ids
        ↓
Encoder (self-attn)
        ↓
Decoder (masked self-attn → cross-attn → FFN)
        ↓
token ids → text like: select <c2> where <c0> = terrence ross
        ↓
sql_parse.py → {sel, agg, conds}
        ↓
readable SQL + sqlite execution
```

## Viva cheat-sheet (study these)

1. **`<cN>`** = pointer to column index N (not the real name)  
2. **Teacher forcing**: decoder input = target[:-1], labels = target[1:]  
3. **Causal mask**: position t cannot see future tokens  
4. **Padding mask**: attention must ignore `<pad>`  
5. **Cross-attn Q/K/V**: Q from decoder, K/V from encoder  
6. **Post-norm**: `LayerNorm(x + sublayer(x))`  
7. **Weight sharing**: token embedding weights tied with output projection  
8. **Beam search**: keep top-4 partial sequences by log-prob sum  

## Status

Code for the full pipeline is in this repo.  
You still need a **full GPU training run** before numbers / demo look good.
