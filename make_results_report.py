"""
make_results_report.py
----------------------
Builds the Section 4 Results tables/figures into results/.

Creates:
  results/table1_data.json / .md
  results/table2_model_training.json / .md
  results/table3_official_metrics.json / .md
  results/table4_component_accuracy.json / .md
  results/RESULTS.md                 (all tables together)
  results/lr_schedule.png            (Figure 3)
  results/loss_curves.png            (Figure 2, if training_history.json exists)

Also fills missing metric slots from existing pred files when possible.

Optional:
  python make_results_report.py --run_missing_evals
      -> runs greedy/dev and beam/test if those pred files are missing
        (can take a long time on CPU)
"""

import argparse
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import sentencepiece as spm
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from starter.dataset import get_default_paths
from starter.data_prep import clean_text
from sql_parse import sql_logical_equal
from model.transformer import Transformer, count_parameters


RESULTS = os.path.join(_ROOT, "results")
MAX_SRC_LEN = 512
MAX_TGT_LEN = 128


def ensure_results_dir():
    if not os.path.exists(RESULTS):
        os.makedirs(RESULTS)


def count_jsonl(path):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() != "":
                n = n + 1
    return n


def length_stats_for_split(pairs_path, sp, max_src, max_tgt):
    """
    Compute mean/max source & target token lengths.
    Also count how many would be dropped if max length limits were applied.
    """
    src_lens = []
    tgt_lens = []
    dropped = 0

    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            row = json.loads(line)
            src_ids = sp.EncodeAsIds(row["src"])
            tgt_ids = sp.EncodeAsIds(row["tgt"])
            # target in training also has bos/eos (+2)
            tgt_full = len(tgt_ids) + 2

            src_lens.append(len(src_ids))
            tgt_lens.append(tgt_full)

            if len(src_ids) > max_src or tgt_full > max_tgt:
                dropped = dropped + 1

    def mean(xs):
        if len(xs) == 0:
            return 0.0
        return float(sum(xs)) / float(len(xs))

    return {
        "pairs": len(src_lens),
        "mean_source_len": round(mean(src_lens), 2),
        "max_source_len": max(src_lens) if src_lens else 0,
        "mean_target_len": round(mean(tgt_lens), 2),
        "max_target_len": max(tgt_lens) if tgt_lens else 0,
        "pairs_dropped_too_long": dropped,
        "max_src_threshold": max_src,
        "max_tgt_threshold": max_tgt,
    }


def build_table1(sp):
    processed = os.path.join(_ROOT, "starter", "processed")
    splits = ["train", "dev", "test"]
    table = {}
    for split in splits:
        path = os.path.join(processed, split + "_pairs.jsonl")
        if not os.path.exists(path):
            table[split] = {"error": "missing " + path}
            continue
        print("Computing length stats for", split, "...")
        table[split] = length_stats_for_split(path, sp, MAX_SRC_LEN, MAX_TGT_LEN)
    return table


def build_table2():
    ckpt_path = os.path.join(RESULTS, "best_model.pt")
    history_path = os.path.join(RESULTS, "training_history.json")
    meta_path = os.path.join(RESULTS, "training_meta.json")

    out = {
        "trainable_parameters": None,
        "epochs_trained": None,
        "best_epoch": None,
        "best_dev_loss": None,
        "training_time": None,
        "gpu": None,
        "notes": [],
    }

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
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
        out["trainable_parameters"] = count_parameters(model)
        out["epochs_trained"] = ckpt.get("total_epochs", ckpt.get("epoch"))
        out["best_epoch"] = ckpt.get("epoch")
        out["best_dev_loss"] = ckpt.get("best_dev", ckpt.get("dev_loss"))
    else:
        out["notes"].append("best_model.pt not found")

    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        # find epoch with lowest dev loss
        best_e = None
        best_loss = None
        i = 0
        while i < len(history):
            d = float(history[i]["dev_loss"])
            if best_loss is None or d < best_loss:
                best_loss = d
                best_e = int(history[i]["epoch"])
            i = i + 1
        if best_e is not None:
            out["best_epoch"] = best_e
            out["best_dev_loss"] = best_loss
        out["epochs_trained"] = len(history)
    else:
        out["notes"].append(
            "training_history.json missing (loss curve needs a re-train with updated train.py, or paste Colab logs)."
        )

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        hours = meta.get("total_training_time_hours")
        secs = meta.get("total_training_time_sec")
        gpu = meta.get("gpu_name") or meta.get("device")
        if hours is not None:
            out["training_time"] = str(hours) + " hours"
        elif secs is not None:
            out["training_time"] = str(round(float(secs) / 3600.0, 3)) + " hours"
        out["gpu"] = gpu
    else:
        # create editable template for the student
        template = {
            "device": "cuda",
            "gpu_name": "EDIT ME (e.g. Tesla T4)",
            "total_training_time_sec": None,
            "total_training_time_hours": None,
            "note": "Fill this file if Colab did not save training_meta.json",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2)
        out["training_time"] = "EDIT results/training_meta.json"
        out["gpu"] = "EDIT results/training_meta.json"
        out["notes"].append("Created training_meta.json template for you to fill.")

    return out


def where_equal(a_conds, b_conds):
    set_a = set()
    for col, op, val in a_conds:
        set_a.add((col, op, str(val).lower()))
    set_b = set()
    for col, op, val in b_conds:
        set_b.add((col, op, str(val).lower()))
    return set_a == set_b


def component_accuracy_from_preds(split, pred_path):
    gold_path = os.path.join(_ROOT, "WikiSQL", "data", split + ".jsonl")
    golds = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() != "":
                golds.append(json.loads(line))

    preds = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() != "":
                preds.append(json.loads(line))

    n = min(len(golds), len(preds))
    parse_fail = 0
    sel_ok = 0
    agg_ok = 0
    where_ok = 0
    lf_ok = 0

    i = 0
    while i < n:
        g = golds[i]
        gold = {
            "sel": g["sql"]["sel"],
            "agg": g["sql"]["agg"],
            "conds": [[c[0], c[1], clean_text(c[2])] for c in g["sql"]["conds"]],
        }
        p = preds[i]
        if "error" in p or "query" not in p:
            parse_fail = parse_fail + 1
        else:
            pred = p["query"]
            if pred["sel"] == gold["sel"]:
                sel_ok = sel_ok + 1
            if pred["agg"] == gold["agg"]:
                agg_ok = agg_ok + 1
            if where_equal(pred["conds"], gold["conds"]):
                where_ok = where_ok + 1
            if sql_logical_equal(pred, gold):
                lf_ok = lf_ok + 1
        i = i + 1

    def pct(x):
        return round(100.0 * x / max(n, 1), 2)

    return {
        "n": n,
        "parse_failures_pct": pct(parse_fail),
        "sel_column_correct_pct": pct(sel_ok),
        "agg_correct_pct": pct(agg_ok),
        "where_clause_correct_pct": pct(where_ok),
        "internal_lf_pct": pct(lf_ok),
    }


def load_metrics_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_table3():
    """
    Collect official metrics for:
      Dev greedy, Dev beam(4), Test (prefer beam)
    """
    rows = {
        "dev_greedy": load_metrics_json(os.path.join(RESULTS, "metrics_dev_greedy.json")),
        "dev_beam4": load_metrics_json(os.path.join(RESULTS, "metrics_dev.json")),
        "test_beam4": load_metrics_json(os.path.join(RESULTS, "metrics_test.json")),
        "test_greedy": load_metrics_json(os.path.join(RESULTS, "metrics_test_greedy.json")),
    }

    # if metrics_dev.json exists but method not stored as greedy, treat as beam
    table = []

    def row(split, decoding, metrics, pred_name):
        if metrics is None:
            # try component parse fail from preds if available
            pred_path = os.path.join(RESULTS, pred_name)
            parse_fail = None
            if os.path.exists(pred_path):
                comp = component_accuracy_from_preds(split, pred_path)
                parse_fail = comp["parse_failures_pct"]
            return {
                "split": split,
                "decoding": decoding,
                "logical_form_pct": None,
                "execution_pct": None,
                "parse_failures_pct": parse_fail,
                "status": "MISSING - run evaluate_model.py",
            }
        lf = metrics.get("official_lf_accuracy")
        ex = metrics.get("official_ex_accuracy")
        pf = metrics.get("parse_failure_rate")
        if pf is None and os.path.exists(os.path.join(RESULTS, pred_name)):
            pf = component_accuracy_from_preds(split, os.path.join(RESULTS, pred_name))[
                "parse_failures_pct"
            ]
        return {
            "split": split,
            "decoding": decoding,
            "logical_form_pct": None if lf is None else round(100.0 * float(lf), 2),
            "execution_pct": None if ex is None else round(100.0 * float(ex), 2),
            "parse_failures_pct": pf,
            "status": "ok",
        }

    table.append(row("dev", "greedy", rows["dev_greedy"], "preds_dev_greedy.jsonl"))
    table.append(row("dev", "beam (4)", rows["dev_beam4"], "preds_dev.jsonl"))
    # prefer test beam; if only greedy exists show that
    if rows["test_beam4"] is not None:
        table.append(row("test", "beam (4)", rows["test_beam4"], "preds_test.jsonl"))
    else:
        table.append(row("test", "beam (4) or greedy", rows["test_greedy"], "preds_test.jsonl"))

    return table


def build_table4():
    pred_path = os.path.join(RESULTS, "preds_dev.jsonl")
    if not os.path.exists(pred_path):
        return {"error": "preds_dev.jsonl missing. Run evaluate_model.py first."}
    return component_accuracy_from_preds("dev", pred_path)


def plot_lr_schedule(out_path, d_model=256, warmup=4000, max_step=20000):
    steps = []
    lrs = []
    step = 1
    while step <= max_step:
        arg1 = step ** (-0.5)
        arg2 = step * (warmup ** (-1.5))
        lr = (d_model ** (-0.5)) * min(arg1, arg2)
        steps.append(step)
        lrs.append(lr)
        step = step + 1

    plt.figure(figsize=(8, 4))
    plt.plot(steps, lrs)
    plt.xlabel("Training step")
    plt.ylabel("Learning rate")
    plt.title("Transformer LR schedule (warmup=%d, d_model=%d)" % (warmup, d_model))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved", out_path)


def plot_loss_curves(history_path, out_path):
    if not os.path.exists(history_path):
        # make a placeholder figure explaining what's missing
        plt.figure(figsize=(8, 4))
        plt.text(
            0.5,
            0.5,
            "training_history.json not found.\n"
            "Re-run training with updated train.py\n"
            "OR paste epoch losses into that file.",
            ha="center",
            va="center",
            fontsize=12,
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print("Saved placeholder", out_path)
        return False

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = []
    train_losses = []
    dev_losses = []
    i = 0
    while i < len(history):
        epochs.append(int(history[i]["epoch"]))
        train_losses.append(float(history[i]["train_loss"]))
        dev_losses.append(float(history[i]["dev_loss"]))
        i = i + 1

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, dev_losses, label="dev loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and dev loss per epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved", out_path)
    return True


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print("Wrote", path)


def md_escape(x):
    if x is None:
        return "—"
    return str(x)


def write_results_md(table1, table2, table3, table4):
    path = os.path.join(RESULTS, "RESULTS.md")
    lines = []
    lines.append("# Results (Section 4)")
    lines.append("")
    lines.append("## Table 1 – Data")
    lines.append("")
    lines.append("| | Train | Dev | Test |")
    lines.append("|---|---:|---:|---:|")

    def cell(split, key):
        if split not in table1 or "error" in table1[split]:
            return "—"
        return md_escape(table1[split].get(key))

    lines.append(
        "| Pairs | %s | %s | %s |"
        % (cell("train", "pairs"), cell("dev", "pairs"), cell("test", "pairs"))
    )
    lines.append(
        "| Mean source length (tokens) | %s | %s | %s |"
        % (
            cell("train", "mean_source_len"),
            cell("dev", "mean_source_len"),
            cell("test", "mean_source_len"),
        )
    )
    lines.append(
        "| Max source length (tokens) | %s | %s | %s |"
        % (
            cell("train", "max_source_len"),
            cell("dev", "max_source_len"),
            cell("test", "max_source_len"),
        )
    )
    lines.append(
        "| Mean target length (tokens) | %s | %s | %s |"
        % (
            cell("train", "mean_target_len"),
            cell("dev", "mean_target_len"),
            cell("test", "mean_target_len"),
        )
    )
    lines.append(
        "| Max target length (tokens) | %s | %s | %s |"
        % (
            cell("train", "max_target_len"),
            cell("dev", "max_target_len"),
            cell("test", "max_target_len"),
        )
    )
    lines.append(
        "| Pairs dropped as too long (>512 src or >128 tgt) | %s | %s | %s |"
        % (
            cell("train", "pairs_dropped_too_long"),
            cell("dev", "pairs_dropped_too_long"),
            cell("test", "pairs_dropped_too_long"),
        )
    )
    lines.append("")
    lines.append("## Table 2 – Model and training")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append("| Trainable parameters | %s |" % md_escape(table2.get("trainable_parameters")))
    lines.append("| Epochs trained / best epoch | %s / %s |" % (
        md_escape(table2.get("epochs_trained")),
        md_escape(table2.get("best_epoch")),
    ))
    lines.append("| Best dev loss | %s |" % md_escape(table2.get("best_dev_loss")))
    lines.append("| Training time | %s |" % md_escape(table2.get("training_time")))
    lines.append("| GPU | %s |" % md_escape(table2.get("gpu")))
    if table2.get("notes"):
        lines.append("")
        lines.append("Notes:")
        for n in table2["notes"]:
            lines.append("- " + n)
    lines.append("")
    lines.append("## Table 3 – Official metrics")
    lines.append("")
    lines.append("| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |")
    lines.append("|---|---|---:|---:|---:|")
    i = 0
    while i < len(table3):
        r = table3[i]
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (
                r["split"],
                r["decoding"],
                md_escape(r["logical_form_pct"]),
                md_escape(r["execution_pct"]),
                md_escape(r["parse_failures_pct"]),
            )
        )
        i = i + 1
    lines.append("")
    lines.append("## Table 4 – Component accuracy (dev, beam)")
    lines.append("")
    if "error" in table4:
        lines.append(table4["error"])
    else:
        lines.append("| Component | Accuracy (%) |")
        lines.append("|---|---:|")
        lines.append("| sel column correct | %s |" % table4["sel_column_correct_pct"])
        lines.append("| agg correct | %s |" % table4["agg_correct_pct"])
        lines.append("| WHERE clause correct | %s |" % table4["where_clause_correct_pct"])
    lines.append("")
    lines.append("## Figures checklist")
    lines.append("")
    lines.append("| # | Figure | File |")
    lines.append("|---|---|---|")
    lines.append("| 1 | Positional-encoding heat-map | `positional_encoding_heatmap.png` |")
    lines.append("| 2 | Train/dev loss per epoch | `loss_curves.png` |")
    lines.append("| 3 | Learning-rate schedule | `lr_schedule.png` |")
    lines.append("| 4 | Cross-attention map | `cross_attention_heatmap.png` |")
    lines.append("| 5 | Front-end screenshot | `frontend_screenshot.png` (add manually) |")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Wrote", path)


def run_eval(split, method, out_metrics_name, pred_name):
    """
    Run evaluate_model.py and rename outputs to method-specific names.
    Protects existing beam files from being overwritten.
    """
    ckpt = os.path.join(RESULTS, "best_model.pt")

    # backup default outputs if they exist (usually beam)
    src_metrics = os.path.join(RESULTS, "metrics_" + split + ".json")
    src_preds = os.path.join(RESULTS, "preds_" + split + ".jsonl")
    bak_metrics = src_metrics + ".bak"
    bak_preds = src_preds + ".bak"

    if os.path.exists(src_metrics):
        with open(src_metrics, "r", encoding="utf-8") as f:
            data = f.read()
        with open(bak_metrics, "w", encoding="utf-8") as f:
            f.write(data)
    if os.path.exists(src_preds):
        with open(src_preds, "r", encoding="utf-8") as f:
            data = f.read()
        with open(bak_preds, "w", encoding="utf-8") as f:
            f.write(data)

    cmd = [
        sys.executable,
        os.path.join(_ROOT, "evaluate_model.py"),
        "--checkpoint",
        ckpt,
        "--split",
        split,
        "--method",
        method,
        "--beam_size",
        "4",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=_ROOT, check=False)

    # save method-specific copies from fresh outputs
    if os.path.exists(src_metrics):
        with open(src_metrics, "r", encoding="utf-8") as f:
            data = f.read()
        with open(os.path.join(RESULTS, out_metrics_name), "w", encoding="utf-8") as f:
            f.write(data)
        print("Saved", out_metrics_name)

    if os.path.exists(src_preds):
        with open(src_preds, "r", encoding="utf-8") as f:
            data = f.read()
        with open(os.path.join(RESULTS, pred_name), "w", encoding="utf-8") as f:
            f.write(data)
        print("Saved", pred_name)

    # restore beam/default backups if we were not writing the default beam names
    if out_metrics_name != ("metrics_" + split + ".json") and os.path.exists(bak_metrics):
        with open(bak_metrics, "r", encoding="utf-8") as f:
            data = f.read()
        with open(src_metrics, "w", encoding="utf-8") as f:
            f.write(data)
    if pred_name != ("preds_" + split + ".jsonl") and os.path.exists(bak_preds):
        with open(bak_preds, "r", encoding="utf-8") as f:
            data = f.read()
        with open(src_preds, "w", encoding="utf-8") as f:
            f.write(data)


def maybe_run_missing_evals():
    # Dev greedy
    if not os.path.exists(os.path.join(RESULTS, "metrics_dev_greedy.json")):
        print("\n=== Running DEV greedy eval (this can take a while) ===")
        run_eval("dev", "greedy", "metrics_dev_greedy.json", "preds_dev_greedy.jsonl")
    else:
        print("DEV greedy metrics already exist.")

    # Test beam
    if not os.path.exists(os.path.join(RESULTS, "metrics_test.json")):
        print("\n=== Running TEST beam eval (this can take a long time) ===")
        run_eval("test", "beam", "metrics_test.json", "preds_test.jsonl")
    else:
        print("TEST beam metrics already exist.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_missing_evals",
        action="store_true",
        help="Also run greedy/dev and beam/test if metrics are missing.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ensure_results_dir()

    _, _, sp_model = get_default_paths()
    if not os.path.exists(sp_model):
        print("Tokenizer missing. Run starter/tokenizer.py first.")
        return
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model)

    if args.run_missing_evals:
        maybe_run_missing_evals()

    print("\nBuilding Table 1...")
    table1 = build_table1(sp)
    save_json(os.path.join(RESULTS, "table1_data.json"), table1)

    print("\nBuilding Table 2...")
    table2 = build_table2()
    save_json(os.path.join(RESULTS, "table2_model_training.json"), table2)

    print("\nBuilding Table 3...")
    table3 = build_table3()
    save_json(os.path.join(RESULTS, "table3_official_metrics.json"), table3)

    print("\nBuilding Table 4...")
    table4 = build_table4()
    save_json(os.path.join(RESULTS, "table4_component_accuracy.json"), table4)

    print("\nMaking figures...")
    plot_lr_schedule(os.path.join(RESULTS, "lr_schedule.png"))
    plot_loss_curves(
        os.path.join(RESULTS, "training_history.json"),
        os.path.join(RESULTS, "loss_curves.png"),
    )

    # remind about positional / cross-attn / frontend
    if not os.path.exists(os.path.join(RESULTS, "positional_encoding_heatmap.png")):
        print("Missing positional_encoding_heatmap.png — run starter/check_starter.py")
    if not os.path.exists(os.path.join(RESULTS, "cross_attention_heatmap.png")):
        print("Missing cross_attention_heatmap.png — run evaluate_model.py --make_attn_plot")
    if not os.path.exists(os.path.join(RESULTS, "frontend_screenshot.png")):
        print("Missing frontend_screenshot.png — take a Streamlit screenshot and save it there.")

    write_results_md(table1, table2, table3, table4)
    print("\nDone. Open results/RESULTS.md")


if __name__ == "__main__":
    main()
