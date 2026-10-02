"""
evaluate_model.py
-----------------
1) Run our model and write predictions in WikiSQL official format
2) Call Salesforce WikiSQL/evaluate.py for lf_accuracy + ex_accuracy
3) Also report parse / sel / agg / where breakdowns
4) Optional cross-attention plot

Official pred line format:
  {"query": {"sel": ..., "agg": ..., "conds": [...]}}
or
  {"error": "parse failed: ..."}

Official command (run from WikiSQL/):
  python evaluate.py data/dev.jsonl data/dev.db ../results/preds_dev.jsonl
"""

import argparse
import json
import os
import subprocess
import sys

import torch
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from starter.dataset import get_default_paths
from starter.data_prep import make_source, load_tables, clean_text
from decode import load_model, load_tokenizer, decode_one
from sql_parse import parse_generated_sql, sql_logical_equal


def load_gold_examples(jsonl_path, tables_path):
    """Load WikiSQL examples in file order (must match official evaluate)."""
    tables = load_tables(tables_path)
    examples = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            row = json.loads(line)
            table_id = row["table_id"]
            headers = tables.get(table_id, [])
            examples.append(
                {
                    "question": row["question"],
                    "sql": row["sql"],
                    "table_id": table_id,
                    "headers": headers,
                    "src": make_source(row["question"], headers) if len(headers) > 0 else "",
                }
            )
    return examples


def where_equal(a_conds, b_conds):
    set_a = set()
    for col, op, val in a_conds:
        set_a.add((col, op, str(val).lower()))
    set_b = set()
    for col, op, val in b_conds:
        set_b.add((col, op, str(val).lower()))
    return set_a == set_b


def normalize_gold_sql(sql):
    """Make gold cond values lowercase strings like our parser output."""
    conds = []
    for c in sql["conds"]:
        conds.append([c[0], c[1], clean_text(c[2])])
    return {
        "sel": sql["sel"],
        "agg": sql["agg"],
        "conds": conds,
    }


@torch.no_grad()
def save_cross_attention_plot(model, sp, example, device, out_path):
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    src_ids = sp.EncodeAsIds(example["src"])
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    enc_out, src_mask = model.encode(src)
    ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    max_len = 64
    t = 0
    while t < max_len:
        logits = model.decode_step(ys, enc_out, src_mask)
        next_id = torch.argmax(logits[:, -1, :], dim=-1)
        ys = torch.cat([ys, next_id.unsqueeze(0)], dim=1)
        if int(next_id.item()) == eos_id:
            break
        t = t + 1

    _, attns = model(src, ys, return_attn=True)
    cross = attns["dec_cross_attn"][-1][0]
    avg = cross.mean(dim=0).detach().cpu().numpy()

    src_pieces = sp.EncodeAsPieces(example["src"])
    src_labels = []
    for p in src_pieces:
        src_labels.append(p.replace("\u2581", "▁"))

    tgt_ids = ys[0].tolist()
    tgt_labels = []
    i = 0
    while i < len(tgt_ids):
        piece = sp.IdToPiece(tgt_ids[i])
        tgt_labels.append(piece.replace("\u2581", "▁"))
        i = i + 1

    avg = avg[: len(tgt_labels), : len(src_labels)]

    plt.figure(figsize=(max(8, len(src_labels) * 0.25), max(4, len(tgt_labels) * 0.35)))
    plt.imshow(avg, aspect="auto", cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(src_labels)), src_labels, rotation=90, fontsize=7)
    plt.yticks(range(len(tgt_labels)), tgt_labels, fontsize=8)
    plt.xlabel("Source tokens")
    plt.ylabel("Generated tokens")
    plt.title("Decoder cross-attention (last layer, avg heads)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved attention plot:", out_path)


def write_predictions(examples, model, sp, device, method, beam_size, pred_path):
    """
    Write one official pred JSON per example (same order as source jsonl).
    Also return breakdown counters.
    """
    n = len(examples)
    parse_fail = 0
    sel_ok = 0
    agg_ok = 0
    where_ok = 0
    lf_ok = 0

    out = open(pred_path, "w", encoding="utf-8")

    i = 0
    while i < n:
        ex = examples[i]
        gold = normalize_gold_sql(ex["sql"])

        if ex["src"] == "":
            pred_line = {"error": "missing table headers"}
            parse_fail = parse_fail + 1
        else:
            decoded = decode_one(
                model,
                sp,
                ex["src"],
                device,
                method=method,
                beam_size=beam_size,
                headers=ex["headers"],
            )
            pred = decoded["sql_dict"]

            if pred is None:
                pred_line = {"error": "parse failed: " + str(decoded["parse_error"])}
                parse_fail = parse_fail + 1
            else:
                # official evaluator expects this shape
                pred_line = {"query": pred}

                if pred["sel"] == gold["sel"]:
                    sel_ok = sel_ok + 1
                if pred["agg"] == gold["agg"]:
                    agg_ok = agg_ok + 1
                if where_equal(pred["conds"], gold["conds"]):
                    where_ok = where_ok + 1
                if sql_logical_equal(pred, gold):
                    lf_ok = lf_ok + 1

        out.write(json.dumps(pred_line, ensure_ascii=False) + "\n")

        if (i + 1) % 100 == 0 or (i + 1) == n:
            print("  wrote preds", i + 1, "/", n)

        i = i + 1

    out.close()

    breakdown = {
        "n": n,
        "parse_fail": parse_fail,
        "sel_ok": sel_ok,
        "agg_ok": agg_ok,
        "where_ok": where_ok,
        "lf_ok_internal": lf_ok,
    }
    return breakdown


def run_official_evaluate(source_file, db_file, pred_file):
    """
    Call Salesforce WikiSQL/evaluate.py
    Must run with cwd=WikiSQL so `import lib...` works.
    """
    wiki_dir = os.path.join(_ROOT, "WikiSQL")
    eval_script = os.path.join(wiki_dir, "evaluate.py")

    # evaluate.py args are positional and usually relative to WikiSQL/
    # Pass absolute paths to be safe with spaces in folder names.
    cmd = [
        sys.executable,
        eval_script,
        source_file,
        db_file,
        pred_file,
    ]

    print("\nRunning official WikiSQL evaluator:")
    print(" ", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=wiki_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("Official evaluator failed.")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None

    # evaluate.py prints a JSON object at the end
    stdout = result.stdout.strip()
    print(stdout)

    # find the JSON block (last {...})
    start = stdout.rfind("{")
    end = stdout.rfind("}")
    if start < 0 or end < 0:
        print("Could not parse official evaluator output.")
        return None

    metrics = json.loads(stdout[start : end + 1])
    return metrics


def maybe_make_limited_source(source_file, limit, out_path):
    """If limit>0, write a truncated source jsonl for fair official eval."""
    if limit <= 0:
        return source_file

    with open(source_file, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        n = 0
        for line in f_in:
            if line.strip() == "":
                continue
            f_out.write(line if line.endswith("\n") else line + "\n")
            n = n + 1
            if n >= limit:
                break
    return out_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="results/best_model.pt")
    p.add_argument("--split", type=str, default="dev", choices=["dev", "test", "train"])
    p.add_argument("--limit", type=int, default=0, help="If >0, only first N examples")
    p.add_argument("--method", type=str, default="beam", choices=["beam", "greedy"])
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--make_attn_plot", action="store_true")
    p.add_argument(
        "--skip_model",
        action="store_true",
        help="Only run official evaluate.py on an existing pred file",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(_ROOT, "WikiSQL", "data")
    results_dir = os.path.join(_ROOT, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    source_file = os.path.join(data_dir, args.split + ".jsonl")
    tables_file = os.path.join(data_dir, args.split + ".tables.jsonl")
    db_file = os.path.join(data_dir, args.split + ".db")
    pred_file = os.path.join(results_dir, "preds_" + args.split + ".jsonl")

    # optional limited source so official eval line counts match
    if args.limit > 0:
        limited_source = os.path.join(results_dir, "source_" + args.split + "_limit.jsonl")
        source_for_eval = maybe_make_limited_source(source_file, args.limit, limited_source)
    else:
        source_for_eval = source_file

    if not args.skip_model:
        ckpt_path = args.checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(_ROOT, ckpt_path)

        if not os.path.exists(ckpt_path):
            print("Checkpoint not found:", ckpt_path)
            print("Train first: python train.py")
            return

        _, _, sp_model = get_default_paths()
        sp = load_tokenizer(sp_model)
        model, _ = load_model(ckpt_path, device)

        examples = load_gold_examples(source_file, tables_file)
        if args.limit > 0:
            examples = examples[: args.limit]

        print("Generating predictions for", len(examples), "examples...")
        breakdown = write_predictions(
            examples, model, sp, device, args.method, args.beam_size, pred_file
        )

        def pct(x):
            return 100.0 * x / max(breakdown["n"], 1)

        print("\n===== INTERNAL BREAKDOWN =====")
        print("examples           :", breakdown["n"])
        print("parse failure rate : %.2f%%" % pct(breakdown["parse_fail"]))
        print("select-col acc     : %.2f%%" % pct(breakdown["sel_ok"]))
        print("aggregation acc    : %.2f%%" % pct(breakdown["agg_ok"]))
        print("where-clause acc   : %.2f%%" % pct(breakdown["where_ok"]))
        print("Wrote predictions  :", pred_file)

        if args.make_attn_plot and len(examples) > 0 and examples[0]["src"] != "":
            plot_path = os.path.join(results_dir, "cross_attention_heatmap.png")
            save_cross_attention_plot(model, sp, examples[0], device, plot_path)
    else:
        if not os.path.exists(pred_file):
            print("Missing pred file:", pred_file)
            return
        breakdown = None

    # ----- official Salesforce evaluator -----
    official = run_official_evaluate(source_for_eval, db_file, pred_file)
    if official is None:
        return

    print("\n===== OFFICIAL WIKISQL evaluate.py =====")
    print("logical-form accuracy (lf_accuracy):", official.get("lf_accuracy"))
    print("execution accuracy   (ex_accuracy):", official.get("ex_accuracy"))

    metrics_path = os.path.join(results_dir, "metrics_" + args.split + ".json")
    metrics = {
        "split": args.split,
        "method": args.method,
        "official_lf_accuracy": official.get("lf_accuracy"),
        "official_ex_accuracy": official.get("ex_accuracy"),
        "pred_file": pred_file,
        "source_file": source_for_eval,
        "db_file": db_file,
    }
    if breakdown is not None:
        metrics["parse_failure_rate"] = 100.0 * breakdown["parse_fail"] / max(breakdown["n"], 1)
        metrics["select_accuracy"] = 100.0 * breakdown["sel_ok"] / max(breakdown["n"], 1)
        metrics["aggregation_accuracy"] = 100.0 * breakdown["agg_ok"] / max(breakdown["n"], 1)
        metrics["where_accuracy"] = 100.0 * breakdown["where_ok"] / max(breakdown["n"], 1)
        metrics["n"] = breakdown["n"]

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Wrote", metrics_path)


if __name__ == "__main__":
    main()
