"""
train.py
--------
Train the Text-to-SQL Transformer from scratch.

Manual settings:
  - teacher forcing
  - CrossEntropy + label_smoothing=0.1, ignore <pad>
  - Adam(beta1=0.9, beta2=0.98, eps=1e-9)
  - LR = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5), warmup=4000
  - batch_size=64, epochs=20
  - save checkpoint with lowest DEV loss

RESUME / GOOGLE DRIVE
---------------------
Checkpoints are saved to --out_dir (use a Drive path in Colab).

By default, if results/last_model.pt already exists, training CONTINUES
from that epoch (so a broken Colab session can restart safely).

Examples:
  # first run / continue automatically
  python train.py --epochs 20 --batch_size 64 --out_dir "/content/drive/MyDrive/GenAI_Project2/results"

  # force restart from scratch
  python train.py --epochs 20 --batch_size 64 --out_dir ".../results" --no_resume
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from starter.dataset import make_dataloader, get_default_paths
from model.transformer import Transformer, count_parameters


class NoamOpt:
    """Original Transformer learning-rate schedule wrapper around Adam."""

    def __init__(self, d_model, warmup, optimizer):
        self.d_model = d_model
        self.warmup = warmup
        self.optimizer = optimizer
        self._step = 0
        self._rate = 0.0

    def step(self):
        self._step = self._step + 1
        rate = self.rate()
        for pg in self.optimizer.param_groups:
            pg["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup ** (-1.5))
        return (self.d_model ** (-0.5)) * min(arg1, arg2)

    def state_dict(self):
        return {
            "step": self._step,
            "rate": self._rate,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        self._step = int(state.get("step", 0))
        self._rate = float(state.get("rate", 0.0))
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        # keep Adam param group lr in sync
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.rate()


def make_criterion(pad_id, label_smoothing=0.1):
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )
    return criterion


def resolve_out_dir(out_dir):
    """
    If out_dir is absolute (e.g. Google Drive path), use it as-is.
    If relative, put it under the project root.
    """
    if os.path.isabs(out_dir):
        return out_dir
    return os.path.join(_ROOT, out_dir)


def run_epoch(model, loader, criterion, optimizer, device, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        decoder_in = tgt[:, :-1]
        labels = tgt[:, 1:]

        if train_mode:
            optimizer.zero_grad()

        if train_mode:
            logits = model(src, decoder_in)
        else:
            with torch.no_grad():
                logits = model(src, decoder_in)

        vocab_size = logits.size(-1)
        loss = criterion(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
        )

        if train_mode:
            loss.backward()
            optimizer.step()

        n_tokens = int((labels != model.pad_id).sum().item())
        total_loss = total_loss + float(loss.item()) * max(n_tokens, 1)
        total_tokens = total_tokens + max(n_tokens, 1)
        n_batches = n_batches + 1

    avg = total_loss / max(total_tokens, 1)
    return avg, n_batches


def save_checkpoint(
    path,
    model,
    sp,
    args,
    epoch,
    train_loss,
    dev_loss,
    optimizer,
    best_dev,
):
    """
    Save everything needed to RESUME training later:
      - model weights
      - optimizer + Noam step
      - next epoch to run
      - best_dev so far
    """
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "vocab_size": model.vocab_size,
        "pad_id": model.pad_id,
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_encoder_layers": args.n_encoder_layers,
        "n_decoder_layers": args.n_decoder_layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "epoch": epoch,                 # last finished epoch
        "next_epoch": epoch + 1,        # where to continue
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "best_dev": best_dev,
        "step": optimizer._step,
        "total_epochs": args.epochs,
    }

    # write to a temp file first, then rename (safer if session dies mid-save)
    tmp_path = path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)
    print("Saved checkpoint:", path)


def try_load_resume(last_path, model, optimizer, device):
    """
    Load last_model.pt if present.
    Returns: (start_epoch, best_dev, loaded_bool)
    """
    if not os.path.exists(last_path):
        return 1, float("inf"), False

    print("Found checkpoint, resuming from:", last_path)
    ckpt = torch.load(last_path, map_location=device)

    model.load_state_dict(ckpt["model_state"])

    if "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    else:
        # older checkpoint without optimizer: at least restore step count
        optimizer._step = int(ckpt.get("step", 0))
        for pg in optimizer.optimizer.param_groups:
            pg["lr"] = optimizer.rate()
        print("Warning: old checkpoint had no optimizer state; LR step restored only.")

    start_epoch = int(ckpt.get("next_epoch", ckpt.get("epoch", 0) + 1))
    best_dev = float(ckpt.get("best_dev", ckpt.get("dev_loss", float("inf"))))

    print(
        "Resume info: next_epoch=%d | step=%d | best_dev=%.4f"
        % (start_epoch, optimizer._step, best_dev)
    )
    return start_epoch, best_dev, True


def append_history(out_dir, record):
    """
    Append one epoch record to training_history.json
    (needed for loss / LR plots in the results report).
    """
    path = os.path.join(out_dir, "training_history.json")
    history = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    # replace same epoch if resumed/re-run
    new_history = []
    i = 0
    while i < len(history):
        if int(history[i].get("epoch", -1)) != int(record["epoch"]):
            new_history.append(history[i])
        i = i + 1
    new_history.append(record)

    # sort by epoch
    def epoch_key(item):
        return int(item.get("epoch", 0))

    new_history.sort(key=epoch_key)

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(new_history, f, indent=2)
    os.replace(tmp, path)


def write_training_meta(out_dir, device, total_time_sec, gpu_name):
    path = os.path.join(out_dir, "training_meta.json")
    meta = {
        "device": str(device),
        "gpu_name": gpu_name,
        "total_training_time_sec": total_time_sec,
        "total_training_time_hours": round(total_time_sec / 3600.0, 3) if total_time_sec else None,
    }
    # keep old total if we are resuming and adding time
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                old = json.load(f)
            prev = float(old.get("total_training_time_sec", 0) or 0)
            meta["total_training_time_sec"] = prev + float(total_time_sec or 0)
            meta["total_training_time_hours"] = round(
                meta["total_training_time_sec"] / 3600.0, 3
            )
            if not gpu_name:
                meta["gpu_name"] = old.get("gpu_name")
            if not meta["device"]:
                meta["device"] = old.get("device")
        except Exception:
            pass

    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_encoder_layers", type=int, default=3)
    p.add_argument("--n_decoder_layers", type=int, default=3)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=4000)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument(
        "--max_train_batches",
        type=int,
        default=0,
        help="If >0, only use this many train batches per epoch (debug).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Folder for checkpoints. Use a Google Drive path in Colab.",
    )
    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume from out_dir/last_model.pt if it exists (default).",
    )
    p.add_argument(
        "--no_resume",
        dest="resume",
        action="store_false",
        help="Ignore existing checkpoints and train from scratch.",
    )
    p.set_defaults(resume=True)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_pairs, dev_pairs, sp_model = get_default_paths()
    if not os.path.exists(train_pairs) or not os.path.exists(sp_model):
        print("Missing processed data or tokenizer.")
        print("Run: python starter/data_prep.py")
        print("Then: python starter/tokenizer.py")
        return

    train_loader, train_ds = make_dataloader(
        train_pairs, sp_model, batch_size=args.batch_size, shuffle=True
    )
    dev_loader, dev_ds = make_dataloader(
        dev_pairs, sp_model, batch_size=args.batch_size, shuffle=False
    )

    sp = train_ds.sp
    pad_id = train_ds.pad_id
    vocab_size = sp.GetPieceSize()

    model = Transformer(
        vocab_size=vocab_size,
        pad_id=pad_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    model = model.to(device)

    print("Vocab size:", vocab_size)
    print("Train examples:", len(train_ds))
    print("Dev examples:", len(dev_ds))
    print("Parameters:", count_parameters(model))

    base_opt = torch.optim.Adam(
        model.parameters(),
        lr=0.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    optimizer = NoamOpt(args.d_model, args.warmup, base_opt)
    criterion = make_criterion(pad_id, label_smoothing=args.label_smoothing)

    out_dir = resolve_out_dir(args.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Created out_dir:", out_dir)
    else:
        print("Using out_dir:", out_dir)

    best_path = os.path.join(out_dir, "best_model.pt")
    last_path = os.path.join(out_dir, "last_model.pt")

    start_epoch = 1
    best_dev = float("inf")
    run_time_sec = 0.0

    gpu_name = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)

    if args.resume:
        start_epoch, best_dev, loaded = try_load_resume(
            last_path, model, optimizer, device
        )
        if loaded and start_epoch > args.epochs:
            print(
                "Checkpoint already finished all %d epochs. Nothing to do."
                % args.epochs
            )
            print("Best model path:", best_path)
            return
        if not loaded:
            print("No last_model.pt found. Starting from scratch.")
    else:
        print("--no_resume set. Starting from scratch.")

    epoch = start_epoch
    while epoch <= args.epochs:
        t0 = time.time()
        print("\n===== Epoch %d / %d =====" % (epoch, args.epochs))

        if args.max_train_batches > 0:
            limited = []
            n = 0
            for b in train_loader:
                limited.append(b)
                n = n + 1
                if n >= args.max_train_batches:
                    break

            model.train()
            total_loss = 0.0
            total_tokens = 0
            for batch in limited:
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)
                decoder_in = tgt[:, :-1]
                labels = tgt[:, 1:]
                optimizer.zero_grad()
                logits = model(src, decoder_in)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )
                loss.backward()
                optimizer.step()
                n_tokens = int((labels != pad_id).sum().item())
                total_loss = total_loss + float(loss.item()) * max(n_tokens, 1)
                total_tokens = total_tokens + max(n_tokens, 1)
            train_loss = total_loss / max(total_tokens, 1)
        else:
            train_loss, _ = run_epoch(
                model, train_loader, criterion, optimizer, device, train_mode=True
            )

        dev_loss, _ = run_epoch(
            model, dev_loader, criterion, optimizer, device, train_mode=False
        )

        dt = time.time() - t0
        run_time_sec = run_time_sec + dt
        print(
            "Epoch %d/%d | train_loss=%.4f | dev_loss=%.4f | lr=%.6f | time=%.1fs"
            % (epoch, args.epochs, train_loss, dev_loss, optimizer._rate, dt)
        )

        # log for results plots
        append_history(
            out_dir,
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "dev_loss": float(dev_loss),
                "lr": float(optimizer._rate),
                "step": int(optimizer._step),
                "epoch_time_sec": float(dt),
            },
        )

        # always save LAST so we can resume after disconnect
        save_checkpoint(
            last_path,
            model,
            sp,
            args,
            epoch,
            train_loss,
            dev_loss,
            optimizer,
            best_dev if dev_loss >= best_dev else dev_loss,
        )

        if dev_loss < best_dev:
            best_dev = dev_loss
            save_checkpoint(
                best_path,
                model,
                sp,
                args,
                epoch,
                train_loss,
                dev_loss,
                optimizer,
                best_dev,
            )
            print("  -> new best dev loss = %.4f" % best_dev)

        epoch = epoch + 1

    write_training_meta(out_dir, device, run_time_sec, gpu_name)

    print("\nTraining finished.")
    print("Last checkpoint :", last_path)
    print("Best checkpoint :", best_path)
    print("History file    :", os.path.join(out_dir, "training_history.json"))


if __name__ == "__main__":
    main()
