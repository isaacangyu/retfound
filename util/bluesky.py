"""Compare benchmark runs against BlueSky, the APTOS 2021 (2nd APTOS Big Data Competition) winner.

Reference: Zhang et al., "Predicting Diabetic Macular Edema Treatment Responses Using OCT: Dataset and
Methods of APTOS Competition", Medical Image Analysis 109 (2026) / arXiv 2505.05768, Tables 4-5.
BlueSky's final score was 0.7440 (mean of its 14 stage-1 + stage-2 indices).

Stages are kept apart: stage 1 is compared with our test eyes from TrainingSet1 (AHA-0000..0250),
stage 2 with those from TrainingSet2 (AHA-0251..0500).
Our predictions are scored with the challenge's rules, per eye (slices of one eye are averaged):
  binary targets (IRF, SRF, PED, HRF, continue injection, and the pre-treatment flags): AUC
  preCST, CST: share of eyes predicted within +-7.5% of the true value
  VA (logMAR): within +-0.05 when the true VA < 1, else within +-7.5%
The challenge did not score the pre-treatment flags (and has no stage-1 PED or stage-2 HRF), so those
show an empty BlueSky bar.

Per run, in benchmark/<run>/:
  vs_bluesky.tsv                 every target x stage: our score, BlueSky's score, gap
  vs_bluesky_stage{1,2}.png      bar chart, ours vs BlueSky, all targets
  roc_stage{1,2}.png             ROC curve per binary target (empty panel when it can't be drawn)
Over all runs, in benchmark/: vs_bluesky.tsv and vs_bluesky_stage{1,2}.png (best run per target).
Regenerate everything with:  python -m util.bluesky [benchmark_dir]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# BlueSky per-index scores as printed in Table 5. The paper's text and table disagree on some cells
# (the text calls 0.7026 BlueSky's stage-1 CST; the table prints it under CI), so these are the table's.
BLUESKY = {
    1: {"preCST": 0.5643, "VA": 0.3216, "CI": 0.7026, "CST": 0.5906, "IRF": 0.9293, "SRF": 0.9218, "HRF": 0.8593},
    2: {"preCST": 0.6871, "VA": 0.5234, "CI": 0.7828, "CST": 0.6930, "IRF": 0.9338, "SRF": 0.9264, "PED": 0.9796},
}
BLUESKY_FINAL = 0.7440
STAGE_SETS = {1: ("TrainingSet1", 0, 250), 2: ("TrainingSet2", 251, 500)}  # patient-number range per stage

# our column key (predictions_test.csv) -> (challenge index name, metric)
INDEX = {
    "preCST": ("preCST", "within_7.5pct"), "CST": ("CST", "within_7.5pct"), "VA": ("VA", "va_rule"),
    "continue_injection": ("CI", "auc"),
    "IRF": ("IRF", "auc"), "SRF": ("SRF", "auc"), "PED": ("PED", "auc"), "HRF": ("HRF", "auc"),
    "preIRF": ("preIRF", "auc"), "preSRF": ("preSRF", "auc"), "prePED": ("prePED", "auc"), "preHRF": ("preHRF", "auc"),
}
ORDER = list(dict.fromkeys(i for i, _ in INDEX.values()))
METRIC_LABEL = {"auc": "AUC", "within_7.5pct": "within ±7.5%", "va_rule": "within ±0.05 / ±7.5%"}

# reference palette (dataviz skill): slot 1 blue = ours, slot 2 orange = BlueSky
OURS, BLUE_SKY = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"


def per_eye(pred):
    """Average slice predictions per eye (the challenge scores eyes); targets are per eye already."""
    keys = [k for k in ("patient", "eye") if k in pred]
    num = pred.select_dtypes("number").columns.drop(["image_idx", "n_slices"], errors="ignore")
    return pred.groupby(keys, as_index=False)[list(num)].mean() if keys else pred


def stage_of(patient):
    num = int(str(patient).split("-")[-1])
    return next((s for s, (_, lo, hi) in STAGE_SETS.items() if lo <= num <= hi), None)


def target_arrays(eyes, key):
    p = eyes[f"prob_{key}"] if f"prob_{key}" in eyes else eyes[f"pred_{key}"]
    y, p = eyes[f"target_{key}"].to_numpy(float), p.to_numpy(float)
    ok = ~(np.isnan(y) | np.isnan(p))
    return y[ok], p[ok]


def challenge_score(metric, y, p):
    if not len(y):
        return np.nan
    if metric == "auc":
        return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    if metric == "within_7.5pct":
        return float(np.mean(np.abs(p - y) <= 0.075 * np.abs(y)))
    tol = np.where(y < 1, 0.05, 0.075 * np.abs(y))  # va_rule
    return float(np.mean(np.abs(p - y) <= tol))


def load_eyes(run_dir):
    path = os.path.join(run_dir, "predictions_test.csv")
    if not os.path.isfile(path):
        return None
    eyes = per_eye(pd.read_csv(path))
    eyes["stage"] = eyes["patient"].map(stage_of)
    return eyes


def run_targets(eyes):
    keys = [k for k in INDEX if f"target_{k}" in eyes]
    return sorted(keys, key=lambda k: ORDER.index(INDEX[k][0]))


def score_run(run_dir):
    """One row per (stage, target) with our score on that stage's eyes and BlueSky's."""
    eyes = load_eyes(run_dir)
    if eyes is None:
        return pd.DataFrame()
    run = os.path.basename(os.path.normpath(run_dir))
    rows = []
    for stage in STAGE_SETS:
        sub = eyes[eyes.stage == stage]
        for key in run_targets(eyes):
            index, metric = INDEX[key]
            y, p = target_arrays(sub, key)
            ours, ref = challenge_score(metric, y, p), BLUESKY[stage].get(index, np.nan)
            rows.append({"run": run, "stage": stage, "train_set": STAGE_SETS[stage][0], "index": index,
                         "metric": metric, "n_eyes": len(y), "n_pos": int(y.sum()) if metric == "auc" else pd.NA,
                         "ours": ours, "bluesky": ref, "gap": ours - ref})
    return pd.DataFrame(rows)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(labelsize=8, colors=INK_2, length=0)


def _footer(fig, height, text):
    fig.text(0.02, 0.06 / height, text, fontsize=7, color=INK_2, ha="left", va="bottom")


NOTE = (f"BlueSky = APTOS 2021 winner (final {BLUESKY_FINAL}); Zhang et al., Med. Image Anal. 2026, Table 5. "
        "Ours: test eyes from the same training set.\nBlueSky was scored on the challenge's own test set, "
        "so this is indicative only. Empty bar = no score.")


def plot_bars(df, path, title, ours_label="This run"):
    """Horizontal bars per target: ours (blue) and BlueSky (orange); an outlined empty bar = no score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(df)
    top, bottom = 1.05, 0.95  # inches: two-line title + legend; x-axis + footnote
    height = top + bottom + 0.6 * max(n, 2)
    fig, ax = plt.subplots(figsize=(7.5, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ys, h = np.arange(n)[::-1], 0.34
    for y, (_, r) in zip(ys, df.iterrows()):
        for dy, v, color in ((h / 2 + 0.02, r.ours, OURS), (-h / 2 - 0.02, r.bluesky, BLUE_SKY)):
            if np.isnan(v):  # empty bar spanning the axis
                ax.barh(y + dy, 1, height=h, facecolor="none", edgecolor=color, lw=1, ls=(0, (3, 2)), alpha=0.6)
                ax.text(0.01, y + dy, "no score", va="center", fontsize=7, color=MUTED)
            else:
                ax.barh(y + dy, v, height=h, color=color, edgecolor=SURFACE, lw=1)
                ax.text(v + 0.01, y + dy, f"{v:.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(ys, [f"{r['index']}  ({METRIC_LABEL[r.metric]}, n={r.n_eyes})" for _, r in df.iterrows()],
                  fontsize=8.5, color=INK)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Challenge score (higher is better)", fontsize=8.5, color=INK_2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=OURS, label=ours_label),
               plt.Rectangle((0, 0), 1, 1, color=BLUE_SKY, label="BlueSky")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
              fontsize=7.5, labelcolor=INK_2)
    fig.text(0.02, 1 - 0.12 / height, title, fontsize=10, color=INK, fontweight="bold", ha="left", va="top")
    _footer(fig, height, NOTE)
    fig.subplots_adjust(left=0.3, right=0.97, top=1 - top / height, bottom=bottom / height)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_roc(eyes, stage, path, title):
    """One ROC panel per binary target on this stage's eyes; empty panel when one class only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in run_targets(eyes) if INDEX[k][1] == "auc"]
    if not keys:
        return
    sub = eyes[eyes.stage == stage]
    cols = min(len(keys), 4)
    rows = int(np.ceil(len(keys) / cols))
    top, bottom = 0.7, 0.75
    height = top + bottom + 2.6 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols + 0.6, height), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    for ax, key in zip(axes.flat, keys):
        _style(ax)
        index = INDEX[key][0]
        y, p = target_arrays(sub, key)
        ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ref = BLUESKY[stage].get(index, np.nan)
        lines = []
        if len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=OURS, lw=2)
            lines.append(f"ours AUC {roc_auc_score(y, p):.2f}")
        else:
            why = "no eyes" if not len(y) else f"all {int(y[0])}s"
            ax.text(0.5, 0.5, f"no curve\n({why} in test, n={len(y)})", ha="center", va="center",
                    fontsize=7.5, color=MUTED)
        lines.append(f"BlueSky AUC {ref:.2f}" if not np.isnan(ref) else "BlueSky: no score")
        ax.text(0.97, 0.04, "\n".join(lines), ha="right", va="bottom", fontsize=7, color=INK_2)
        ax.set_title(f"{index}  (n={len(y)}, pos={int(y.sum())})", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("False positive rate", fontsize=7.5, color=INK_2)
        ax.set_ylabel("True positive rate", fontsize=7.5, color=INK_2)
    fig.text(0.02, 1 - 0.12 / height, title, fontsize=10, color=INK, fontweight="bold", ha="left", va="top")
    _footer(fig, height, "Per eye. BlueSky reported AUC only (no curve); dashed = chance.")
    fig.subplots_adjust(left=0.08, right=0.98, top=1 - top / height, bottom=bottom / height, wspace=0.35, hspace=0.45)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _clear_old(folder):
    for f in ("vs_bluesky.png",):  # single-image layout from before stages were split
        p = os.path.join(folder, f)
        if os.path.isfile(p):
            os.remove(p)


def write_run(run_dir):
    df = score_run(run_dir)
    if df.empty:
        return df
    from util.oct_tasks import write_aligned_tsv
    _clear_old(run_dir)
    write_aligned_tsv(df, os.path.join(run_dir, "vs_bluesky.tsv"))
    run = os.path.basename(os.path.normpath(run_dir))
    eyes = load_eyes(run_dir)
    for stage, (tset, _, _) in STAGE_SETS.items():
        plot_bars(df[df.stage == stage], os.path.join(run_dir, f"vs_bluesky_stage{stage}.png"),
                  f"{run}\nvs BlueSky stage {stage} ({tset} eyes)")
        plot_roc(eyes, stage, os.path.join(run_dir, f"roc_stage{stage}.png"),
                 f"{run}: ROC, stage {stage} ({tset} eyes)")
    return df


def write_summary(benchmark_dir):
    """All runs in one table; the plots show the best run per target, one per stage."""
    paths = sorted(glob.glob(os.path.join(benchmark_dir, "*", "predictions_test.csv")))
    runs = [r for r in (score_run(os.path.dirname(p)) for p in paths) if not r.empty]
    if not runs:
        return
    from util.oct_tasks import write_aligned_tsv
    _clear_old(benchmark_dir)
    df = pd.concat(runs, ignore_index=True)
    write_aligned_tsv(df, os.path.join(benchmark_dir, "vs_bluesky.tsv"))
    for stage, (tset, _, _) in STAGE_SETS.items():
        s = df[df.stage == stage]
        best = (s.assign(_s=s.ours.fillna(-1)).sort_values("_s", ascending=False)
                 .drop_duplicates("index").drop(columns="_s")
                 .sort_values("index", key=lambda c: c.map(ORDER.index)))
        plot_bars(best, os.path.join(benchmark_dir, f"vs_bluesky_stage{stage}.png"),
                  f"Best run per target vs BlueSky stage {stage} ({tset} eyes)\n"
                  "(which run: benchmark/vs_bluesky.tsv)", ours_label="Our best run")


def update(run_dir, benchmark_dir):
    """Called after each benchmark; a plotting error never fails the run."""
    try:
        write_run(run_dir)
        write_summary(benchmark_dir)
    except Exception as e:
        print(f"[BlueSky] comparison skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    bdir = sys.argv[1] if len(sys.argv) > 1 else "benchmark"
    for p in sorted(glob.glob(os.path.join(bdir, "*", "predictions_test.csv"))):
        write_run(os.path.dirname(p))
    write_summary(bdir)
    print(f"Wrote vs_bluesky.tsv, vs_bluesky_stage{{1,2}}.png, roc_stage{{1,2}}.png in {bdir}/ and each run folder")
