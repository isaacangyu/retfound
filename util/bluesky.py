"""Compare benchmark runs against BlueSky, the APTOS 2021 (2nd APTOS Big Data Competition) winner.

Reference: Zhang et al., "Predicting Diabetic Macular Edema Treatment Responses Using OCT: Dataset and
Methods of APTOS Competition", Medical Image Analysis 109 (2026) / arXiv 2505.05768, Tables 4-5.
BlueSky's final score was 0.7440 (mean of the 14 indices below).

Our predictions are scored with the challenge's rules, per eye (slices of one eye are averaged):
  IRF, SRF, PED, HRF, continue injection (CI): AUC
  preCST, CST: share of eyes predicted within +-7.5% of the true value
  VA (logMAR): within +-0.05 when the true VA < 1, else within +-7.5%
Pre-treatment IRF/SRF/PED/HRF were not scored in the challenge, so they have no reference.

Writes benchmark/<run>/vs_bluesky.{tsv,png} per run and benchmark/vs_bluesky.{tsv,png} over all runs.
Regenerate everything with:  python -m util.bluesky [benchmark_dir]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# BlueSky per-index scores as printed in Table 5. Stage 1 = preliminary test set, stage 2 = final
# test set. The paper's text and table disagree on some cells (the text calls 0.7026 BlueSky's
# stage-1 CST, the table prints it under CI), so these are the table's values.
BLUESKY = {  # index: (stage 1, stage 2)
    "preCST": (0.5643, 0.6871),
    "VA":     (0.3216, 0.5234),
    "CI":     (0.7026, 0.7828),
    "CST":    (0.5906, 0.6930),
    "IRF":    (0.9293, 0.9338),
    "SRF":    (0.9218, 0.9264),
    "PED":    (np.nan, 0.9796),
    "HRF":    (0.8593, np.nan),
}
BLUESKY_FINAL = 0.7440

# our column key (predictions_test.csv) -> (challenge index, metric)
INDEX = {
    "preCST": ("preCST", "within_7.5pct"),
    "CST": ("CST", "within_7.5pct"),
    "VA": ("VA", "va_rule"),
    "continue_injection": ("CI", "auc"),
    "IRF": ("IRF", "auc"), "SRF": ("SRF", "auc"), "PED": ("PED", "auc"), "HRF": ("HRF", "auc"),
}
ORDER = ["preCST", "CST", "VA", "CI", "IRF", "SRF", "PED", "HRF"]
METRIC_LABEL = {"auc": "AUC", "within_7.5pct": "within ±7.5%", "va_rule": "within ±0.05 / ±7.5%"}

# reference palette (dataviz skill): slot 1 blue = ours, slot 2 orange = BlueSky final, gray = context
OURS, BLUE_SKY, CONTEXT = "#2a78d6", "#eb6834", "#8a8984"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def per_eye(pred):
    """Average slice predictions per eye (the challenge scores eyes); targets are per eye already."""
    keys = [k for k in ("patient", "eye") if k in pred]
    num = pred.select_dtypes("number").columns.drop(["image_idx", "n_slices"], errors="ignore")
    return pred.groupby(keys, as_index=False)[list(num)].mean() if keys else pred


def challenge_score(metric, y, p):
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    if not len(y):
        return np.nan, 0
    if metric == "auc":
        return (float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan), len(y)
    if metric == "within_7.5pct":
        return float(np.mean(np.abs(p - y) <= 0.075 * np.abs(y))), len(y)
    tol = np.where(y < 1, 0.05, 0.075 * np.abs(y))  # va_rule
    return float(np.mean(np.abs(p - y) <= tol)), len(y)


def score_run(run_dir):
    """Rows (one per challenge index this run predicts) with our score and BlueSky's."""
    path = os.path.join(run_dir, "predictions_test.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    eyes = per_eye(pd.read_csv(path))
    rows = []
    for key, (index, metric) in INDEX.items():
        if f"target_{key}" not in eyes:
            continue
        p = eyes[f"prob_{key}"] if f"prob_{key}" in eyes else eyes[f"pred_{key}"]
        score, n = challenge_score(metric, eyes[f"target_{key}"].to_numpy(float), p.to_numpy(float))
        s1, s2 = BLUESKY[index]
        rows.append({"run": os.path.basename(os.path.normpath(run_dir)), "index": index, "metric": metric,
                     "n_eyes": n, "ours": score, "bluesky_stage1": s1, "bluesky_stage2": s2,
                     "gap_vs_stage2": score - s2})
    rows.sort(key=lambda r: ORDER.index(r["index"]))
    return pd.DataFrame(rows)


def plot(df, path, title, subtitle, ours_label="This run"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(df)
    top, bottom = 0.85, 0.95  # inches for title + legend, and x-axis + footnote
    height = top + bottom + 0.55 * max(n, 2)
    fig, ax = plt.subplots(figsize=(7.5, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ys = np.arange(n)[::-1]
    for y, (_, r) in zip(ys, df.iterrows()):
        vals = [v for v in (r.ours, r.bluesky_stage2) if not np.isnan(v)]
        if len(vals) == 2:  # connector shows the gap
            ax.plot(vals, [y, y], color=GRID, lw=2, zorder=1, solid_capstyle="round")
        if not np.isnan(r.bluesky_stage1):
            ax.scatter(r.bluesky_stage1, y, s=60, facecolor="none", edgecolor=CONTEXT, lw=1.5, zorder=2)
        if not np.isnan(r.bluesky_stage2):
            ax.scatter(r.bluesky_stage2, y, s=80, marker="D", color=BLUE_SKY, edgecolor=SURFACE, lw=2, zorder=3)
        if np.isnan(r.ours):
            ax.text(0.01, y, "n/a (one class in test)", va="center", fontsize=8, color=INK_2)
        else:
            ax.scatter(r.ours, y, s=80, color=OURS, edgecolor=SURFACE, lw=2, zorder=4)
            ax.annotate(f"{r.ours:.2f}", (r.ours, y), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
    labels = [f"{r['index']}  ({METRIC_LABEL[r.metric]}, n={r.n_eyes})" for _, r in df.iterrows()]
    ax.set_yticks(ys, labels, fontsize=8.5, color=INK)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Challenge score (higher is better)", fontsize=8.5, color=INK_2)
    ax.tick_params(axis="x", labelsize=8, colors=INK_2, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=8, color=OURS, label=ours_label),
        plt.Line2D([], [], ls="", marker="D", ms=7, color=BLUE_SKY, label="BlueSky, final test (stage 2)"),
        plt.Line2D([], [], ls="", marker="o", ms=7, mfc="none", mec=CONTEXT, mew=1.5,
                   label="BlueSky, preliminary (stage 1)"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, frameon=False,
              fontsize=7.5, labelcolor=INK_2, handletextpad=0.3, columnspacing=1.2)
    fig.text(0.02, 1 - 0.12 / height, title, fontsize=10, color=INK, fontweight="bold", ha="left", va="top")
    fig.text(0.02, 0.06 / height, subtitle, fontsize=7, color=INK_2, ha="left", va="bottom")
    fig.subplots_adjust(left=0.3, right=0.97, top=1 - top / height, bottom=bottom / height)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


NOTE = (f"BlueSky (APTOS 2021 winner, final score {BLUESKY_FINAL}): Zhang et al., Med. Image Anal. 2026, "
        "Table 5. Ours: test split, per eye.\nBlueSky was scored on the challenge's own test sets, "
        "so the comparison is indicative only.")


def write_run(run_dir):
    df = score_run(run_dir)
    if df.empty:
        return df
    from util.oct_tasks import write_aligned_tsv
    write_aligned_tsv(df, os.path.join(run_dir, "vs_bluesky.tsv"))
    plot(df, os.path.join(run_dir, "vs_bluesky.png"),
         f"{os.path.basename(os.path.normpath(run_dir))} vs BlueSky", NOTE)
    return df


def write_summary(benchmark_dir):
    """All runs in one table; the plot shows each index's best run."""
    runs = [score_run(os.path.dirname(p)) for p in sorted(glob.glob(os.path.join(benchmark_dir, "*", "predictions_test.csv")))]
    runs = [r for r in runs if not r.empty]
    if not runs:
        return
    from util.oct_tasks import write_aligned_tsv
    df = pd.concat(runs, ignore_index=True)
    write_aligned_tsv(df, os.path.join(benchmark_dir, "vs_bluesky.tsv"))
    best = (df.assign(_s=df.ours.fillna(-1)).sort_values("_s", ascending=False)
              .drop_duplicates("index").drop(columns="_s"))
    best = best.sort_values("index", key=lambda s: s.map(ORDER.index))
    plot(best, os.path.join(benchmark_dir, "vs_bluesky.png"), "Best run per index vs BlueSky",
         NOTE + " Which run is best per index: benchmark/vs_bluesky.tsv.", ours_label="Our best run")


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
    print(f"Wrote vs_bluesky.{{tsv,png}} in {bdir}/ and each run folder")
