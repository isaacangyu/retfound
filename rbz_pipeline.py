"""RBZ pipeline: APTOS pre features -> synthetic IMAGINE proteomics -> anti-VEGF response class.

1. One row per eye from TrainingAnnotation.csv. By default only eyes with OCT in data/index.csv;
   --incl-no-eyes keeps every CSV eye.
2. preCST/preIRF/preSRF/prePED/preHRF go into gen_imagine_cohort.synthesize_oct_proteomics_cohort, which adds
   radiomic volumes, cytokines (VEGF, IL-6, MCP-1, MMP-1, ANGPTL4) and the ordinal 4-tier syn_responder_class
   (C0 Non < C1 Slow < C2 Early < C3 Super), drawn from Abraham et al. 2022 class-level statistics.
3. A tabular classifier (pre features + synthetic proteomics) predicts syn_responder_class.
   --rbz-post instead predicts the real post-treatment CSV labels with a multi-head torch MLP (see PostMLP).

Splits:
    --rbz / --rbz-post  val = DME eyes treated with ranibizumab (anti-VEGF code 2); train = every other eye of a patient
               with no rbz eye. No test split. That is 3 eyes with OCT, or 8 with --incl-no-eyes.
    (default)  patient-level --split_ratio train/val/test with --split_seed (as util/oct_tasks.select_and_split).

Outputs: benchmark/rbz/<run>/{cohort.csv, predictions_<split>.csv, analysis.tsv, filter.json}
and one summary row per run in benchmark/rbz/analysis.tsv.
Run = rbz_<model>[_incl-no-eyes]... with --rbz, rbz_<model>_random-split... without.
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from gen_imagine_cohort import CLASS_NAMES, synthesize_oct_proteomics_cohort
from util.bluesky import challenge_score
from util.oct_tasks import OctTaskLoss, file_lock, read_aligned_tsv, task_metrics, write_aligned_tsv

CLASSES = [0, 1, 2, 3]
SYNTH_INPUTS = {"preCST": "CST", "preIRF": "IRF", "preSRF": "SRF", "prePED": "PED", "preHRF": "HRF"}
CYTOKINES = ["syn_VEGF_pg_ml", "syn_IL6_pg_ml", "syn_MCP1_pg_ml", "syn_MMP1_pg_ml", "syn_ANGPTL4_pg_ml"]
FEATURES = (["age", "gender", "preVA", "preCST", "preIRF", "preSRF", "prePED", "preHRF",
             "syn_panmacular_vol_mm3", "syn_srf_vol_mm3"] + [f"log_{c}" for c in CYTOKINES])
# Synthetic-class task: real age / gender / preVA are replaced by demographics simulated from the class
# (gen_imagine_cohort, paper Table 1). --rbz-post (real post labels) keeps FEATURES.
CLASS_FEATURES = (["syn_age", "syn_male", "syn_right_eye", "preCST", "preIRF", "preSRF", "prePED", "preHRF",
                   "syn_panmacular_vol_mm3", "syn_srf_vol_mm3"] + [f"log_{c}" for c in CYTOKINES])
RBZ_CODE, DME_CODE = 2, 3
MODELS = ["logreg", "xgb", "rf", "mlp", "svm"]
# --rbz-post targets (post-treatment CSV columns) and their APTOS challenge metric (util/bluesky.py)
POST_TARGETS = [("VA", "real"), ("CST", "real"), ("continue injection", "binary"),
                ("IRF", "binary"), ("SRF", "binary"), ("PED", "binary"), ("HRF", "binary")]
POST_METRIC = {"VA": "va_rule", "CST": "within_7.5pct"}  # every binary target: AUC


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rbz", action="store_true", help="Validate on DME + ranibizumab eyes; train on the rest")
    p.add_argument("--rbz-post", dest="rbz_post", action="store_true",
                   help="rbz split, but predict the post-treatment labels (VA, CST, continue injection, IRF, SRF, "
                        "PED, HRF) with a multi-head MLP and a mixed MSE + BCE loss (implies --model mlp)")
    p.add_argument("--epochs", type=int, default=400, help="--rbz-post: full-batch training epochs")
    p.add_argument("--lr", type=float, default=3e-3, help="--rbz-post: AdamW learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-2, help="--rbz-post: AdamW weight decay")
    p.add_argument("--dropout", type=float, default=0.2, help="--rbz-post: dropout in trunk and heads")
    p.add_argument("--incl-no-eyes", dest="incl_no_eyes", action="store_true",
                   help="Use every eye in the CSV, including eyes with no OCT images in data/index.csv")
    p.add_argument("--model", default=None, choices=MODELS,
                   help="Classifier (one): logreg (default), xgb (XGBoost), rf (random forest), mlp, svm (RBF)")
    p.add_argument("--diagnosis", nargs="+", type=int, default=[DME_CODE], help="Diagnosis codes; 0 = all")
    p.add_argument("--synth_seed", type=int, default=42, help="Seed for gen_imagine_cohort")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--split_ratio", nargs=3, type=float, default=[0.6, 0.2, 0.2])
    p.add_argument("--seed", type=int, default=0, help="Classifier seed")
    p.add_argument("--data_path", default="./data")
    p.add_argument("--csv", default="TrainingAnnotation.csv")
    p.add_argument("--out_dir", default="./benchmark/rbz")
    args = p.parse_args()
    if args.rbz_post:
        if args.model not in (None, "mlp"):
            p.error("--rbz-post uses its own multi-head MLP; drop --model or pass --model mlp")
        args.model, args.rbz = "mlp", True
    args.model = args.model or "logreg"
    return args


def run_name(args):
    name = ("rbz-post_mlp" if args.rbz_post else f"rbz_{args.model}" + ("" if args.rbz else "_random-split")) + ("_incl-no-eyes" if args.incl_no_eyes else "")
    if args.diagnosis != [DME_CODE]:
        name += "_dx" + "-".join(map(str, args.diagnosis))
    if args.synth_seed != 42:
        name += f"_synth{args.synth_seed}"
    return name


def load_eyes(args):
    """One row per eye (CSV row), optionally restricted to eyes with OCT, filtered by diagnosis."""
    df = pd.read_csv(args.csv)
    df["patient ID"] = df["patient ID"].str.strip()
    n_dup = int(df["patient ID"].duplicated().sum())
    if n_dup:
        print(f"[warn] {n_dup} duplicate eye IDs in {args.csv}; keeping the first")
        df = df.drop_duplicates("patient ID")
    df = df.rename(columns={"patient ID": "csv_id"})
    df["patient"] = df.csv_id.str[:9]
    df["eye"] = df.csv_id.str[-1].map({"R": "OD", "L": "OS"})
    df["has_oct"] = df.csv_id.isin(set(pd.read_csv(os.path.join(args.data_path, "index.csv"), usecols=["csv_id"]).csv_id))
    if not args.incl_no_eyes:
        df = df[df.has_oct]
    if 0 not in args.diagnosis:
        df = df[df.diagnosis.isin(args.diagnosis)]
    return df.reset_index(drop=True)


def synthesize(df, seed):
    """Run gen_imagine_cohort on the pre features; eyes missing any synthesizer input are dropped."""
    ok = df[list(SYNTH_INPUTS)].notna().all(axis=1)
    if (~ok).any():
        print(f"[synth] dropped {int((~ok).sum())} eyes missing one of {list(SYNTH_INPUTS)}")
    base = df[ok].reset_index(drop=True)
    cohort = synthesize_oct_proteomics_cohort(base[list(SYNTH_INPUTS)].rename(columns=SYNTH_INPUTS), seed=seed)
    syn = cohort[[c for c in cohort.columns if c.startswith("syn_")]]
    out = pd.concat([base, syn], axis=1)
    for c in CYTOKINES:
        out[f"log_{c}"] = np.log(out[c])
    assert out.syn_responder_class.isin(CLASSES).all()
    return out


def split(df, args):
    """Adds a 'split' column. --rbz: val = DME rbz eyes, train = eyes of other patients. Else patient-level split."""
    if args.rbz:
        val = (df.diagnosis == DME_CODE) & (df["anti-VEGF"] == RBZ_CODE)
        if not val.any():
            raise ValueError("--rbz: no DME + ranibizumab eyes after filtering")
        val_patients = set(df.patient[val])
        df["split"] = np.where(val, "val", np.where(df.patient.isin(val_patients), "excluded", "train"))
        n_ex = int((df.split == "excluded").sum())
        if n_ex:
            print(f"[split] excluded {n_ex} non-rbz eyes of rbz patients (no patient leaks into train)")
        splits = ["train", "val"]
    else:
        patients = sorted(df.patient.unique())
        random.Random(args.split_seed).shuffle(patients)
        n = len(patients)
        n_train, n_val = round(n * args.split_ratio[0]), round(n * args.split_ratio[1])
        group = {p: "train" for p in patients[:n_train]}
        group.update({p: "val" for p in patients[n_train:n_train + n_val]})
        group.update({p: "test" for p in patients[n_train + n_val:]})
        df["split"] = df.patient.map(group)
        splits = ["train", "val", "test"]
    for s in splits:
        if not (df.split == s).any():
            raise ValueError(f"split '{s}' is empty")
    sets = {s: set(df.patient[df.split == s]) for s in splits}
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            assert not sets[a] & sets[b], f"patients in both {a} and {b}: {sorted(sets[a] & sets[b])}"
    return df, splits


class BalancedXGB(BaseEstimator, ClassifierMixin):
    """XGBoost with balanced sample weights; re-encodes labels so classes missing from train are allowed."""

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, y):
        from xgboost import XGBClassifier
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        self.est_ = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=5.0, min_child_weight=2,
                                  gamma=0.1, random_state=self.seed, n_jobs=1, verbosity=0)
        self.est_.fit(X, self.le_.transform(y), sample_weight=compute_sample_weight("balanced", y))
        return self

    def predict_proba(self, X):
        return self.est_.predict_proba(X)


def make_clf(args):
    s = args.seed
    est = {
        # elastic net: C = 1/alpha = 0.5, half L1 / half L2 (features are standardised)
        "logreg": lambda: LogisticRegression(penalty="elasticnet", solver="saga", C=0.5, l1_ratio=0.5,
                                             max_iter=20000, class_weight="balanced", random_state=s),
        "xgb": lambda: BalancedXGB(seed=s),
        "rf": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                             random_state=s, n_jobs=-1),
        # sklearn 1.4 MLP has no sample_weight, so this one is unweighted
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=2000,
                                     early_stopping=False, random_state=s),
        "svm": lambda: SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced",
                           random_state=s),
    }[args.model]()
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", est)])


def predict_proba(model, X):
    """Probabilities over all 4 classes (classes unseen in training get 0)."""
    p = model.predict_proba(X)
    full = np.zeros((len(X), len(CLASSES)))
    full[:, [CLASSES.index(c) for c in model.classes_]] = p
    return full


def metrics(y, prob, split_name):
    yhat = prob.argmax(1)
    aucs = [roc_auc_score(y == c, prob[:, c]) for c in CLASSES if 0 < (y == c).sum() < len(y)]
    kappa = (cohen_kappa_score(y, yhat, weights="quadratic", labels=CLASSES)
             if len(np.unique(np.concatenate([y, yhat]))) > 1 else np.nan)
    r = {"split": split_name, "n": len(y),
         **{f"n_C{c}": int((y == c).sum()) for c in CLASSES},
         "accuracy": accuracy_score(y, yhat),
         "f1_macro": f1_score(y, yhat, labels=CLASSES, average="macro", zero_division=0),
         "kappa_qw": kappa,
         "roc_auc_ovr": float(np.mean(aucs)) if aucs else np.nan,  # macro over classes present with both labels
         "n_auc_classes": len(aucs)}
    parts = [v for v in (r["f1_macro"], r["roc_auc_ovr"], r["kappa_qw"]) if not np.isnan(v)]
    r["score"] = float(np.mean(parts)) if parts else np.nan
    return r


# ---------------------------------------------------------------------------
# Performance plots: one panel per target that is not 100% one class in the split
# ---------------------------------------------------------------------------

def plottable(y, kind):
    """True if the split's labels for this target vary (binary: both classes; real: not constant)."""
    y = y[~np.isnan(y)]
    return len(y) >= 2 and len(np.unique(y)) >= 2 if kind == "binary" else len(y) >= 2 and np.std(y) > 0


def plot_perf(panels, path, title):
    """panels: list of (name, kind, y_true, y_pred). binary -> ROC curve, real -> predicted vs true."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    from util.bluesky import GRID, INK, INK_2, OURS, SURFACE, _style

    if os.path.isfile(path):
        os.remove(path)
    if not panels:
        print(f"[plot] {os.path.basename(path)}: skipped, every target is one class in this split")
        return
    cols = min(len(panels), 4)
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols + 0.4, 2.9 * rows + 0.6), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (name, kind, y, p) in zip(axes.flat, panels):
        ok = ~(np.isnan(y) | np.isnan(p))
        y, p = y[ok], p[ok]
        _style(ax)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        if kind == "binary":
            y = y.astype(int)
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
            ax.plot(fpr, tpr, color=OURS, lw=2)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal")
            ax.set_xlabel("False positive rate", fontsize=7.5, color=INK_2)
            ax.set_ylabel("True positive rate", fontsize=7.5, color=INK_2)
            note = f"AUC {roc_auc_score(y, p):.2f}"
            ax.set_title(f"{name}  (n={len(y)}, pos={int(y.sum())})", fontsize=8.5, color=INK, loc="left")
        else:
            lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
            pad = 0.05 * (hi - lo or 1)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=GRID, lw=1, ls="--")
            ax.scatter(y, p, s=14, color=OURS, alpha=0.75, edgecolors="none")
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect("equal")
            ax.set_xlabel("True", fontsize=7.5, color=INK_2)
            ax.set_ylabel("Predicted", fontsize=7.5, color=INK_2)
            r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
            note = f"R\u00b2 {r2:.2f}\nMAE {np.mean(np.abs(y - p)):.3g}"
            ax.set_title(f"{name}  (n={len(y)})", fontsize=8.5, color=INK, loc="left")
        ax.text(0.97, 0.04, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK_2)
    fig.suptitle(title, x=0.02, ha="left", fontsize=10, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] {path} ({len(panels)} panels)")


# ---------------------------------------------------------------------------
# --rbz-post: multi-head MLP on the real post-treatment labels
# ---------------------------------------------------------------------------

class PostMLP(nn.Module):
    """Shared trunk + one small head per target.

    trunk: Linear(d, 64) - LayerNorm - GELU - Dropout - Linear(64, 64) - LayerNorm - GELU - Dropout
    head (per target): Linear(64, 32) - GELU - Dropout - Linear(32, 1)
    Output = one value per target: a z-scored value for real targets, a BCE logit for binary ones.
    """

    def __init__(self, d_in, targets, hidden=64, head_hidden=32, dropout=0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, head_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1))
            for _ in targets)

    def forward(self, x):
        z = self.trunk(x)
        return torch.cat([h(z) for h in self.heads], dim=1)


def run_post(args, df, splits, d):
    """Train PostMLP on train eyes (fixed epochs, no val-based selection); score every split."""
    cols = [t for t, _ in POST_TARGETS]
    has_label = df[cols].notna().any(axis=1)
    if (~has_label).any():
        print(f"[post] dropped {int((~has_label).sum())} eyes with every post label missing")
    df = df[has_label].reset_index(drop=True)
    tr = df[df.split == "train"]
    norm = {c: {"mean": float(tr[c].mean()), "std": float(tr[c].std()) or 1.0}
            for c, k in POST_TARGETS if k == "real"}

    def targets(f):
        y = f[cols].to_numpy(float)
        for j, (c, k) in enumerate(POST_TARGETS):
            if k == "real":
                y[:, j] = (y[:, j] - norm[c]["mean"]) / norm[c]["std"]
        return torch.tensor(y, dtype=torch.float32)

    prep = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    prep.fit(tr[FEATURES].to_numpy(float))

    def feats(f):
        return torch.tensor(prep.transform(f[FEATURES].to_numpy(float)), dtype=torch.float32)

    torch.manual_seed(args.seed)
    model = PostMLP(len(FEATURES), POST_TARGETS, dropout=args.dropout)
    loss_fn = OctTaskLoss([k for _, k in POST_TARGETS])  # masked MSE (real) + BCE (binary), summed
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    x_tr, y_tr = feats(tr), targets(tr)
    for ep in range(args.epochs):
        model.train()
        loss = loss_fn(model(x_tr), y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if ep % 100 == 0 or ep == args.epochs - 1:
            print(f"[post] epoch {ep:4d} train loss {loss.item():.4f}")

    model.eval()
    keep = ["csv_id", "patient", "eye", "has_oct", "diagnosis", "anti-VEGF"]
    table, results = [], {}
    for s in splits:
        f = df[df.split == s].reset_index(drop=True)
        with torch.no_grad():
            out = model(feats(f)).numpy().astype(float)
        pred, tgt = out.copy(), f[cols].to_numpy(float)
        for j, (c, k) in enumerate(POST_TARGETS):
            pred[:, j] = 1 / (1 + np.exp(-out[:, j])) if k == "binary" else out[:, j] * norm[c]["std"] + norm[c]["mean"]
        rows = task_metrics(pred, tgt, POST_TARGETS, mode="soft")
        plot_perf([(c, k, tgt[:, j], pred[:, j]) for j, (c, k) in enumerate(POST_TARGETS) if plottable(tgt[:, j], k)],
                  os.path.join(d, f"perf_{s}.png"), f"{os.path.basename(d)}: {s} (n={len(f)} eyes)")
        chal = []
        for j, (c, _) in enumerate(POST_TARGETS):
            ok = ~np.isnan(tgt[:, j])
            m = POST_METRIC.get(c, "auc")
            rows[j]["challenge_metric"] = m
            rows[j]["challenge_score"] = challenge_score(m, tgt[ok, j], pred[ok, j])
            chal.append(rows[j]["challenge_score"])
        chal = [v for v in chal if not np.isnan(v)]
        rows[-1]["challenge_score"] = float(np.mean(chal)) if chal else np.nan
        for r in rows:
            r["split"] = s
        table += rows
        results[s] = rows[-1]
        if s != "train":
            out_df = f[keep].copy()
            for j, c in enumerate(cols):
                out_df[f"true_{c}"] = tgt[:, j]
                out_df[f"pred_{c}"] = pred[:, j]
            out_df.to_csv(os.path.join(d, f"predictions_{s}.csv"), index=False, float_format="%.6f")
    torch.save({"model": model.state_dict(), "norm": norm, "features": FEATURES, "targets": POST_TARGETS},
               os.path.join(d, "model.pth"))
    table = pd.DataFrame(table)
    first = ["split", "target", "kind", "n", "n_pos", "challenge_metric", "challenge_score", "score"]
    table = table[first + [c for c in table.columns if c not in first]].dropna(axis=1, how="all")
    return df, table, results, norm


def main():
    args = get_args()
    name = run_name(args)
    d = os.path.join(args.out_dir, name)
    os.makedirs(d, exist_ok=True)

    eyes = load_eyes(args)
    print(f"[load] {len(eyes)} eyes / {eyes.patient.nunique()} patients "
          f"({int(eyes.has_oct.sum())} with OCT; diagnosis {args.diagnosis})")
    df = synthesize(eyes, args.synth_seed)
    df, splits = split(df, args)
    print(f"[features] {FEATURES if args.rbz_post else CLASS_FEATURES}")

    if args.rbz_post:
        return main_post(args, name, d, df, splits)

    tr = df[df.split == "train"]
    model = make_clf(args).fit(tr[CLASS_FEATURES].to_numpy(float), tr.syn_responder_class.to_numpy())

    rows = []
    keep = ["csv_id", "patient", "eye", "has_oct", "diagnosis", "anti-VEGF"]
    for s in splits:
        f = df[df.split == s].reset_index(drop=True)
        prob = predict_proba(model, f[CLASS_FEATURES].to_numpy(float))
        y = f.syn_responder_class.to_numpy()
        rows.append(metrics(y, prob, s))
        ovr = [(CLASS_NAMES[c], "binary", (y == c).astype(float), prob[:, c]) for c in CLASSES]
        plot_perf([q for q in ovr if plottable(q[2], "binary")],
                  os.path.join(d, f"perf_{s}.png"), f"{name}: {s}, one-vs-rest ROC (n={len(f)} eyes)")
        if s != "train":
            pred = f[keep].copy()
            pred["true_class"] = y
            pred["pred_class"] = prob.argmax(1)
            pred["pred_label"] = [CLASS_NAMES[c] for c in pred.pred_class]
            for c in CLASSES:
                pred[f"prob_C{c}"] = prob[:, c]
            pred.to_csv(os.path.join(d, f"predictions_{s}.csv"), index=False, float_format="%.6f")

    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    write_aligned_tsv(table, os.path.join(d, "analysis.tsv"))
    df.to_csv(os.path.join(d, "cohort.csv"), index=False)

    counts = {s: {"eyes": int((df.split == s).sum()), "patients": int(df.patient[df.split == s].nunique()),
                  "with_oct": int(df.has_oct[df.split == s].sum()),
                  "classes": {CLASS_NAMES[c]: int(((df.split == s) & (df.syn_responder_class == c)).sum())
                              for c in CLASSES}}
              for s in splits + (["excluded"] if (df.split == "excluded").any() else [])}
    info = {"run": name, "args": vars(args), "target": "syn_responder_class", "features": CLASS_FEATURES,
            "counts": counts, "patients": {s: sorted(df.patient[df.split == s].unique()) for s in counts}}
    with open(os.path.join(d, "filter.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)

    ev = next(r for r in rows if r["split"] == ("test" if "test" in splits else "val"))
    summary = {"run": name, "model": args.model, "task": "responder", "rbz": args.rbz, "incl_no_eyes": args.incl_no_eyes,
               "diagnosis": ",".join(map(str, args.diagnosis)), "synth_seed": args.synth_seed,
               "eval_split": ev["split"],
               **{f"n_{s}": counts[s]["eyes"] if s in counts else None for s in ("train", "val", "test")},
               **{k: ev[k] for k in ("accuracy", "f1_macro", "kappa_qw", "roc_auc_ovr", "score")}}
    write_summary(args, summary)
    print(f"[out] {d}")


def write_summary(args, summary):
    path = os.path.join(args.out_dir, "analysis.tsv")
    with file_lock(path + ".lock"):
        new = pd.DataFrame([summary])
        if os.path.isfile(path):
            old = read_aligned_tsv(path)
            if "clf" in old.columns:  # rows written before --clf became --model
                old = old.rename(columns={"clf": "model"})
            if "task" not in old.columns or old.task.isna().any():  # rows written before --rbz-post existed
                old["task"] = old.get("task", pd.Series(index=old.index, dtype=object)).fillna("responder")
            new = pd.concat([old[old.run != summary["run"]], new], ignore_index=True)
        write_aligned_tsv(new.astype({f"n_{s}": "Int64" for s in ("train", "val", "test")}), path)


def main_post(args, name, d, df, splits):
    df, table, results, norm = run_post(args, df, splits, d)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    write_aligned_tsv(table, os.path.join(d, "analysis.tsv"))
    df.to_csv(os.path.join(d, "cohort.csv"), index=False)
    counts = {s: {"eyes": int((df.split == s).sum()), "patients": int(df.patient[df.split == s].nunique()),
                  "with_oct": int(df.has_oct[df.split == s].sum()),
                  "labelled": {c: int(df[c][df.split == s].notna().sum()) for c, _ in POST_TARGETS}}
              for s in splits + (["excluded"] if (df.split == "excluded").any() else [])}
    info = {"run": name, "args": vars(args), "targets": POST_TARGETS, "features": FEATURES, "norm": norm,
            "architecture": PostMLP.__doc__, "counts": counts,
            "patients": {s: sorted(df.patient[df.split == s].unique()) for s in counts}}
    with open(os.path.join(d, "filter.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    ev = results["val"]
    write_summary(args, {"run": name, "model": "mlp-multihead", "task": "post", "rbz": True,
                         "incl_no_eyes": args.incl_no_eyes, "diagnosis": ",".join(map(str, args.diagnosis)),
                         "synth_seed": args.synth_seed, "eval_split": "val",
                         **{f"n_{s}": counts[s]["eyes"] if s in counts else None for s in ("train", "val", "test")},
                         "roc_auc_ovr": ev.get("roc_auc"), "challenge_score": ev.get("challenge_score"),
                         "score": ev["score"]})
    print(f"[out] {d}")


if __name__ == "__main__":
    main()
