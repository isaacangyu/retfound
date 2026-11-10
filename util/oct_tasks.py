"""OCT benchmark tasks: sample filtering, patient split, dataset, loss, metrics, outputs.

The sample store is built by prepare_data.py (data/index.csv + data/<sample_id>/...).
Exactly one task flag picks the targets:

    --multi-binary   post IRF, SRF, HRF          3 sigmoid outputs, BCE
    --single-real    post CST                    1 output, MSE on z-scored CST
    --response       post VA, continue injection VA: MSE (z-scored), injection: BCE
    --pre-features   preCST, preIRF, prePED, preHRF (from features.json)
                                                 CST: MSE (z-scored), flags: BCE

Binary targets (BCE in both modes):
    --soft-label  (default) plain 0/1 targets; the prediction is a probability in [0, 1],
                  scored as a probability (AUC, AP, Brier, MAE). No threshold.
    --hard-label  label-smoothed targets (eps = --smoothing); the prediction is thresholded
                  to a 0/1 class, scored as a class (F1, accuracy, precision, recall, + AUC).
Real targets also get "within_5pct": share of predictions within +-5% of the truth
(the challenge's scoring rule). Missing targets (NaN) are masked out of the loss.
"""
import csv
import json
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import pearsonr
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from torch.utils.data import Dataset

# task name -> list of (column in data/index.csv, kind)
TASKS = {
    "multi_binary": [("IRF", "binary"), ("SRF", "binary"), ("HRF", "binary")],
    "single_real": [("CST", "real")],
    "response": [("VA", "real"), ("continue injection", "binary")],
    "pre_features": [("preCST", "real"), ("preIRF", "binary"), ("prePED", "binary"), ("preHRF", "binary")],
}
SPLITS = ("train", "val", "test")


def task_columns(args):
    return TASKS[args.task_type]


def use_mil(args):
    """MIL (one bag of slices per eye) turns on when both eyes, more than one slice index, or 'all' slices are selected."""
    return len(args.eye) > 1 or len(args.image_idx) > 1 or "all" in args.image_idx


BAG_KEYS = ["patient", "eye", "visit"]


def make_bags(frame):
    """Slice rows -> one row per (patient, eye, visit) bag; targets/features come from the eye's CSV row.

    Adds slice_ids (list of sample_ids) and n_slices; sample_id becomes the bag id
    (e.g. AHA-0000_OS_pre) and image_idx the joined slice list ("0-1-2").
    """
    rows = []
    for key, g in frame.sort_values("sample_id").groupby(BAG_KEYS, sort=True):
        r = g.iloc[0].copy()
        r["slice_ids"] = list(g.sample_id)
        r["n_slices"] = len(g)
        r["image_idx"] = "-".join(map(str, g.image_idx))
        r["sample_id"] = "_".join(key)
        rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Selection + split
# ---------------------------------------------------------------------------

def build_filter(args):
    return {
        "eye": list(args.eye),
        "visit": list(args.visit),
        "image_idx": list(args.image_idx),
        "diagnosis": list(args.diagnosis) if args.diagnosis else "all",
        "anti_vegf": list(args.anti_vegf) if args.anti_vegf else "all",
    }


def select_and_split(args):
    """Filter data/index.csv, drop samples with no target, split 60/20/20 by patient.

    Returns a dict with per-split DataFrames, the filter, the split ratios/seed, and
    z-score stats (from the train split) for real-valued targets. Cached on args.
    """
    if getattr(args, "_oct_split", None) is not None:
        return args._oct_split

    index = pd.read_csv(os.path.join(args.data_path, "index.csv"))
    flt = build_filter(args)
    m = index.eye.isin(flt["eye"]) & index.visit.isin(flt["visit"])
    if "all" not in flt["image_idx"]:
        m &= index.image_idx.isin(flt["image_idx"])
    if flt["diagnosis"] != "all":
        m &= index.diagnosis.isin(flt["diagnosis"])
    if flt["anti_vegf"] != "all":
        m &= index["anti-VEGF"].isin(flt["anti_vegf"])
    cols = [c for c, _ in task_columns(args)]
    m &= index[cols].notna().any(axis=1)
    sel = index[m].reset_index(drop=True)
    if sel.empty:
        raise ValueError(f"No samples match filter {flt} with a non-missing target in {cols}")

    patients = sorted(sel.patient.unique())
    random.Random(args.split_seed).shuffle(patients)
    n = len(patients)
    n_train = round(n * args.split_ratio[0])
    n_val = round(n * args.split_ratio[1])
    groups = {"train": patients[:n_train], "val": patients[n_train:n_train + n_val],
              "test": patients[n_train + n_val:]}
    frames = {s: sel[sel.patient.isin(groups[s])].sort_values("sample_id").reset_index(drop=True)
              for s in SPLITS}
    for s in SPLITS:
        if frames[s].empty:
            raise ValueError(f"Split '{s}' is empty ({n} patients after filtering); relax the filter.")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = set(frames[a].patient) & set(frames[b].patient)
        assert not shared, f"patients in both {a} and {b}: {sorted(shared)}"

    mil = use_mil(args)
    bags = {s: make_bags(frames[s]) for s in SPLITS} if mil else None

    norm = {}
    for c, kind in task_columns(args):
        if kind == "real":
            # stats over data points: bags (eyes) under MIL, so eyes with many slices are not over-weighted
            v = (bags if mil else frames)["train"][c].dropna().astype(float)
            norm[c] = {"mean": float(v.mean()), "std": float(v.std(ddof=0)) or 1.0}

    args._oct_split = {"filter": flt, "split_ratio": list(args.split_ratio), "split_seed": args.split_seed,
                       "frames": frames, "patients": groups, "norm": norm, "mil": mil, "bags": bags}
    return args._oct_split


class OctTaskDataset(Dataset):
    """Items are (image, target float32[n_targets]); real targets z-scored, NaN = missing."""

    def __init__(self, frame, data_path, columns, norm, transform=None):
        self.frame = frame
        self.transform = transform
        self.samples = [(os.path.join(data_path, sid), None) for sid in frame.sample_id]
        t = []
        for c, kind in columns:
            v = frame[c].astype(float).to_numpy()
            if kind == "real":
                v = (v - norm[c]["mean"]) / norm[c]["std"]
            t.append(v)
        self.targets = torch.tensor(np.stack(t, axis=1), dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.samples[idx][0], "image.jpg")).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx].clone()


class OctMilDataset(OctTaskDataset):
    """Items are (slices float[k, C, H, W], target); one item per bag (eye at a visit).

    ``frame`` is the bag-level frame from make_bags. Use with mil_collate.
    """

    def __init__(self, frame, data_path, columns, norm, transform=None):
        super().__init__(frame, data_path, columns, norm, transform)
        self.bags = [[os.path.join(data_path, sid) for sid in ids] for ids in frame.slice_ids]

    def __getitem__(self, idx):
        imgs = []
        for d in self.bags[idx]:
            img = Image.open(os.path.join(d, "image.jpg")).convert("RGB")
            imgs.append(self.transform(img) if self.transform is not None else img)
        return torch.stack(imgs), self.targets[idx].clone()


def mil_collate(batch):
    """[(slices[k_i], t_i)] -> ((all slices [sum k_i, C, H, W], bag sizes [B]), targets [B, n_targets])."""
    imgs, tgts = zip(*batch)
    sizes = torch.tensor([x.shape[0] for x in imgs], dtype=torch.long)
    return (torch.cat(imgs), sizes), torch.stack(tgts)


def forward_samples(model, samples, device):
    """Model outputs per data point. MIL batches ((slices, sizes)) are mean-pooled over each bag's slices."""
    if not isinstance(samples, (tuple, list)):
        return model(samples.to(device, non_blocking=True))
    images, sizes = samples
    out = model(images.to(device, non_blocking=True))
    sizes = sizes.to(out.device)
    bag = torch.repeat_interleave(torch.arange(len(sizes), device=out.device), sizes)
    pooled = torch.zeros(len(sizes), out.shape[1], device=out.device, dtype=out.dtype).index_add_(0, bag, out)
    return pooled / sizes.unsqueeze(1).to(out.dtype)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class OctTaskLoss(nn.Module):
    """Per target: masked BCE (binary) or MSE (real, z-scored); summed over targets.

    Each target's loss is averaged over its non-missing entries. ``smoothing`` > 0
    softens binary targets to y * (1 - eps) + eps / 2 (used by --hard-label).
    """

    def __init__(self, kinds, smoothing=0.0):
        super().__init__()
        self.register_buffer("is_binary", torch.tensor([k == "binary" for k in kinds]), persistent=False)
        self.smoothing = smoothing

    def forward(self, logits, targets):
        logits, targets = logits.float(), targets.float()
        is_bin = self.is_binary.to(logits.device)
        mask = ~torch.isnan(targets)
        t = torch.nan_to_num(targets, nan=0.0)
        t_bin = t * (1 - self.smoothing) + self.smoothing / 2 if self.smoothing > 0 else t
        bce = F.binary_cross_entropy_with_logits(logits, t_bin, reduction="none")
        mse = (logits - t) ** 2
        per_elem = torch.where(is_bin, bce, mse) * mask
        per_target = per_elem.sum(0) / mask.sum(0).clamp(min=1)
        return per_target.sum()

    def extra_repr(self):
        return f"binary={self.is_binary.tolist()}, smoothing={self.smoothing}"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

BIN_KEYS = ["roc_auc", "average_precision", "brier", "prob_mae", "f1", "accuracy", "precision", "recall"]
REAL_KEYS = ["mae", "rmse", "r2", "pearson_r", "within_5pct"]


def label_mode(args):
    return "soft" if args.soft_label else "hard"


def label_smoothing(args):
    """Smoothing eps applied to binary training targets (hard labels only)."""
    return 0.0 if args.soft_label else args.smoothing


def decode_outputs(outputs, targets, columns, norm):
    """Model outputs / dataset targets -> (pred, target) in natural units.

    Binary: pred = sigmoid probability. Real: de-normalised value.
    """
    pred, tgt = outputs.astype(np.float64).copy(), targets.astype(np.float64).copy()
    for j, (c, kind) in enumerate(columns):
        if kind == "binary":
            pred[:, j] = 1 / (1 + np.exp(-pred[:, j]))
        else:
            pred[:, j] = pred[:, j] * norm[c]["std"] + norm[c]["mean"]
            tgt[:, j] = tgt[:, j] * norm[c]["std"] + norm[c]["mean"]
    return pred, tgt


def task_metrics(pred, tgt, columns, mode="soft", threshold=0.5):
    """Rows per target, plus an 'overall' row (mean of each metric over targets).

    Binary, soft: probability metrics (AUC, AP, Brier, MAE); score = mean(AUC, 1 - Brier).
    Binary, hard: class metrics at ``threshold`` (F1, acc, precision, recall) + AUC;
                  score = mean(F1, AUC).
    Real: MAE, RMSE, R^2, Pearson r, within_5pct; score = R^2.
    Overall score = mean of the per-target scores (nan parts skipped).
    """
    rows, scores = [], []
    for j, (c, kind) in enumerate(columns):
        valid = ~np.isnan(tgt[:, j])
        y, p = tgt[valid, j], pred[valid, j]
        r = {"target": c, "kind": kind if kind == "real" else f"binary_{mode}", "n": int(valid.sum())}
        if kind == "binary":
            y = y.astype(int)
            both = len(np.unique(y)) == 2
            r["n_pos"] = int(y.sum())
            r["roc_auc"] = float(roc_auc_score(y, p)) if both else np.nan
            if mode == "soft":
                r["average_precision"] = float(average_precision_score(y, p)) if both else np.nan
                r["brier"] = float(np.mean((p - y) ** 2)) if len(y) else np.nan
                r["prob_mae"] = float(np.mean(np.abs(p - y))) if len(y) else np.nan
                part = [v for v in (r["roc_auc"], 1 - r["brier"]) if not np.isnan(v)]
            else:
                yhat = (p >= threshold).astype(int)
                r["f1"] = float(f1_score(y, yhat, zero_division=0)) if len(y) else np.nan
                r["accuracy"] = float(accuracy_score(y, yhat)) if len(y) else np.nan
                r["precision"] = float(precision_score(y, yhat, zero_division=0)) if len(y) else np.nan
                r["recall"] = float(recall_score(y, yhat, zero_division=0)) if len(y) else np.nan
                part = [v for v in (r["f1"], r["roc_auc"]) if not np.isnan(v)]
        else:
            ok = len(y) >= 2
            r["mae"] = float(mean_absolute_error(y, p)) if len(y) else np.nan
            r["rmse"] = float(np.sqrt(np.mean((y - p) ** 2))) if len(y) else np.nan
            r["r2"] = float(r2_score(y, p)) if ok and np.std(y) > 0 else np.nan  # undefined for constant y
            r["pearson_r"] = float(pearsonr(y, p)[0]) if ok and np.std(y) > 0 and np.std(p) > 0 else np.nan
            # challenge rule: correct if |pred - true| <= 5% of |true| (exact match when true == 0)
            r["within_5pct"] = float(np.mean(np.abs(p - y) <= 0.05 * np.abs(y))) if len(y) else np.nan
            part = [] if np.isnan(r["r2"]) else [r["r2"]]
        r["score"] = float(np.mean(part)) if part else np.nan
        scores.append(r["score"])
        rows.append(r)

    kinds = {r["kind"] for r in rows}
    overall = {"target": "overall", "kind": kinds.pop() if len(kinds) == 1 else "mixed",
               "n": max(r["n"] for r in rows)}
    for k in BIN_KEYS + REAL_KEYS:
        vals = [r[k] for r in rows if k in r and not np.isnan(r[k])]
        overall[k] = float(np.mean(vals)) if vals else np.nan
    valid_scores = [s for s in scores if not np.isnan(s)]
    overall["score"] = float(np.mean(valid_scores)) if valid_scores else np.nan
    rows.append(overall)
    return rows


METRIC_COLS = ["target", "kind", "n", "n_pos"] + BIN_KEYS + REAL_KEYS + ["score"]


def format_rows(rows):
    def f(v):
        if isinstance(v, float):
            return "nan" if np.isnan(v) else f"{v:.4f}"
        return str(v)
    widths = {c: max(len(c), *(len(f(r.get(c, ""))) for r in rows)) for c in METRIC_COLS}
    lines = ["  ".join(c.rjust(widths[c]) for c in METRIC_COLS)]
    lines += ["  ".join(f(r.get(c, "")).rjust(widths[c]) for c in METRIC_COLS) for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark outputs
# ---------------------------------------------------------------------------

def run_dir(args):
    d = os.path.join(args.benchmark_dir, args.task)
    os.makedirs(d, exist_ok=True)
    return d


def write_filter_json(args):
    split = select_and_split(args)
    info = {
        "run": args.task,
        "task_type": args.task_type,
        "targets": [{"name": c, "kind": k} for c, k in task_columns(args)],
        "label_mode": label_mode(args),
        "label_smoothing": label_smoothing(args),
        "head": {"type": args.head, "hidden": args.head_hidden, "dropout": args.head_dropout},
        "filter": split["filter"],
        "split_ratio": split["split_ratio"],
        "split_seed": split["split_seed"],
        "target_normalisation": split["norm"],
        "mil": {"enabled": split["mil"], "bag": BAG_KEYS, "pooling": "mean of slice outputs (logits)"},
        "counts": {s: {"samples": len(split["frames"][s]), "patients": len(split["patients"][s]),
                       "eyes": int(split["frames"][s].groupby(["patient", "eye"]).ngroups),
                       **({"bags": len(split["bags"][s])} if split["mil"] else {})}
                   for s in SPLITS},
        "patients": split["patients"],
        "model": args.model,
        "finetune": args.finetune,
        "adaptation": args.adaptation,
        "input_size": args.input_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    with open(os.path.join(run_dir(args), "filter.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def write_test_outputs(args, rows, pred, tgt, frame, best_epoch=None):
    """benchmark/<run>/analysis.tsv, predictions_test.csv, and a row in benchmark/analysis.tsv."""
    d = run_dir(args)
    columns = task_columns(args)
    write_aligned_tsv(pd.DataFrame(rows).reindex(columns=METRIC_COLS).astype({"n": "Int64", "n_pos": "Int64"}),
                      os.path.join(d, "analysis.tsv"))

    keep = ["sample_id", "patient", "eye", "visit", "image_idx"] + (["n_slices"] if "n_slices" in frame else [])
    out = frame[keep].copy().reset_index(drop=True)
    for j, (c, kind) in enumerate(columns):
        key = c.replace(" ", "_")
        if kind == "binary" and not args.soft_label:
            out[f"prob_{key}"] = pred[:, j]
            out[f"pred_{key}"] = (pred[:, j] >= 0.5).astype(int)
        else:  # real target, or soft binary (prediction = probability)
            out[f"pred_{key}"] = pred[:, j]
        out[f"target_{key}"] = frame[c].astype(float).to_numpy()  # original values, no round-trip error
    out.to_csv(os.path.join(d, "predictions_test.csv"), index=False, float_format="%.6f")

    split = select_and_split(args)
    overall = rows[-1]
    summary = {
        "run": args.task, "task_type": args.task_type,
        "label_mode": label_mode(args), "head": args.head,
        "model": args.model, "finetune": args.finetune, "adaptation": args.adaptation,
        "eye": ",".join(split["filter"]["eye"]), "visit": ",".join(split["filter"]["visit"]),
        "image_idx": ",".join(map(str, split["filter"]["image_idx"])),
        "diagnosis": split["filter"]["diagnosis"] if isinstance(split["filter"]["diagnosis"], str)
        else ",".join(map(str, split["filter"]["diagnosis"])),
        "anti_vegf": split["filter"]["anti_vegf"] if isinstance(split["filter"]["anti_vegf"], str)
        else ",".join(map(str, split["filter"]["anti_vegf"])),
        "split_seed": split["split_seed"],
        "mil": split["mil"],  # n_* count data points: bags (eyes) under MIL, slices otherwise
        **{f"n_{s}": len((split["bags"] if split["mil"] else split["frames"])[s]) for s in SPLITS},
        "best_epoch": best_epoch,
        **{k: overall.get(k, np.nan) for k in BIN_KEYS + REAL_KEYS + ["score"]},
    }
    path = os.path.join(args.benchmark_dir, "analysis.tsv")
    new = pd.DataFrame([summary])
    with file_lock(path + ".lock"):  # parallel runs (train.sh --gpus) share this file
        if os.path.isfile(path):
            old = read_aligned_tsv(path)
            old = old[old.run != args.task]  # re-running a task replaces its row
            new = pd.concat([old, new], ignore_index=True)
        write_aligned_tsv(new, path)

        from util import bluesky  # plots vs the APTOS 2021 winner; errors are logged, not raised
        bluesky.update(d, args.benchmark_dir)


def write_aligned_tsv(df, path):
    """Tab-separated, with each cell space-padded to its column width so the columns line up in an editor."""
    rows = [line.split("\t") for line in df.to_csv(sep="\t", index=False, float_format="%.6f").splitlines()]
    widths = [max(len(r[j]) for r in rows) for j in range(len(rows[0]))]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(c.ljust(w) for c, w in zip(r[:-1], widths)) + "\t" + r[-1] + "\n")


def read_aligned_tsv(path):
    """Reads write_aligned_tsv output (or plain TSV): spaces around the tabs are dropped."""
    return pd.read_csv(path, sep=r" *\t *", engine="python")


@contextmanager
def file_lock(lock_path, timeout=120):
    """Cross-process lock via an exclusively created file; a lock older than `timeout` s is treated as stale."""
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock_path) > timeout:
                    os.remove(lock_path)
                    continue
            except OSError:
                pass
            time.sleep(0.2)
    try:
        yield
    finally:
        os.close(fd)
        try:
            os.remove(lock_path)
        except OSError:
            pass


def push_to_hub(args):
    """Upload checkpoint-best.pth + this run's benchmark files to <hub_repo>/<task>/ (needs a write token).

    After the checkpoint upload, its size on the hub is compared with the local file; if they match,
    the local .pth is deleted (unless --hub_keep_local). The small benchmark files are kept.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.hub_repo, private=not args.hub_public, exist_ok=True)
    ckpt = os.path.join(args.output_dir, args.task, "checkpoint-best.pth")
    files = [(ckpt, "checkpoint-best.pth")]
    files += [(os.path.join(run_dir(args), f), f) for f in ("filter.json", "analysis.tsv", "predictions_test.csv",
                                                                "vs_bluesky.tsv", "vs_bluesky_stage1.png", "vs_bluesky_stage2.png",
                                                                "roc_stage1.png", "roc_stage2.png")]
    for local, name in files:
        if os.path.isfile(local):
            api.upload_file(path_or_fileobj=local, path_in_repo=f"{args.task}/{name}", repo_id=args.hub_repo,
                            commit_message=f"{args.task}: {name}")
            print(f"[Hub] uploaded {local} -> {args.hub_repo}/{args.task}/{name}")

    if args.hub_keep_local or not os.path.isfile(ckpt):
        return
    remote = api.get_paths_info(args.hub_repo, [f"{args.task}/checkpoint-best.pth"])
    local_size = os.path.getsize(ckpt)
    if remote and getattr(remote[0], "size", None) == local_size:
        os.remove(ckpt)
        print(f"[Hub] verified upload ({local_size / 1e9:.2f} GB); deleted local {ckpt}")
    else:
        print(f"[Hub] could not verify the uploaded checkpoint; keeping local {ckpt}")
