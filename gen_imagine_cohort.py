import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Class-level statistics from Abraham et al., Am J Ophthalmol 2022 (IMAGINE, n = 24 DME eyes)
# Classes are ordinal: C0 Non < C1 Slow < C2 Early < C3 Super responder.
# -----------------------------------------------------------------------------
CLASS_NAMES = ["C0_Non_Responder", "C1_Slow_Responder", "C2_Early_Responder", "C3_Super_Responder"]

# Observed counts 5 / 6 / 4 / 9 of 24 eyes
PRIOR = np.array([5, 6, 4, 9]) / 24.0

# Baseline mean CST (um). Reported: Super 614, all Responders 557, Non 360.
# Slow and Early are not reported separately; both get the Slow+Early mean (557*19 - 614*9) / 10 = 505.7.
CST_CLASS_MEAN = np.array([360.0, 505.7, 505.7, 614.0])
CST_PAPER_MEAN = float(PRIOR @ CST_CLASS_MEAN)                       # ~516 um
CST_PAPER_SD = 115.0                                                 # IMAGINE DME cohort SD (total)
_between_var = float(PRIOR @ (CST_CLASS_MEAN - CST_PAPER_MEAN) ** 2)  # ~93^2
CST_WITHIN_SD = np.sqrt(CST_PAPER_SD ** 2 - _between_var)            # ~68 um, so the class mixture has SD 115

# Baseline mean cytokine concentrations (pg/mL) per class. Slow / Early are solved from the pooled means:
#   VEGF:  Super 880.0, Super+Early 848.2, all Responders 732.6, Non 245.4  -> Early 776.7, Slow 482.1
#          (check: Slow+Non = (482.1*6 + 245.4*5) / 11 = 374.5, as reported)
#   MCP-1: Super 513.3, Super+Early 486.4, all Responders 406.7, Non 809.5  -> Early 425.9, Slow 234.0
#   ANGPTL4: Super+Early 1677.4 vs Slow+Non 632.0 (no finer split reported)
CYTOKINE_CLASS_MEAN = {
    "VEGF": np.array([245.4, 482.1, 776.7, 880.0]),
    "MCP1": np.array([809.5, 234.0, 425.9, 513.3]),
    "ANGPTL4": np.array([632.0, 632.0, 1677.4, 1677.4]),
}
# Demographics per class (paper Table 1). Early-responder age is printed "10.5 (55.5)" = mean/SD swapped.
AGE_CLASS_MEAN = np.array([66.6, 55.2, 55.5, 47.2])
AGE_CLASS_SD = np.array([5.7, 5.8, 10.5, 7.8])
MALE_CLASS_P = np.array([2 / 5, 5 / 6, 3 / 4, 9 / 9])         # Super = 100% male: females are never C3
RIGHT_EYE_CLASS_P = np.array([3 / 5, 3 / 6, 1 / 4, 3 / 9])

# Within-class log-SDs are not reported in the paper; these are assumptions.
CYTOKINE_LOG_SD = {"VEGF": 0.85, "MCP1": 0.60, "ANGPTL4": 0.80}
CYTOKINE_LLOQ = {"VEGF": 5.0, "MCP1": 20.0, "ANGPTL4": 50.0}


def _cst_class_posterior(cst, cst_mean, cst_sd):
    """P(class | CST): PRIOR x Gaussian class likelihood, with CST matched to the paper's scale by z-score."""
    z = (cst - cst_mean) / cst_sd
    cst_paper = CST_PAPER_MEAN + CST_PAPER_SD * z
    loglik = -0.5 * ((cst_paper[:, None] - CST_CLASS_MEAN[None, :]) / CST_WITHIN_SD) ** 2
    logp = np.log(PRIOR)[None, :] + loglik
    p = np.exp(logp - logp.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def synthesize_oct_proteomics_cohort(df: pd.DataFrame, seed: int = 42, cst_mean: float = None,
                                     cst_sd: float = None) -> pd.DataFrame:
    """Synthesizes an ordinal 4-tier anti-VEGF responder class, IMAGINE-style volumes and cytokine
    concentrations from baseline APTOS OCT biomarkers.

    Expected input columns in df:
        - CST (float): Central Subfield Thickness in microns
        - IRF, HRF, PED, SRF (0/1): fluid / foci / detachment presence

    Generative order:
        1. class ~ P(class | CST) = PRIOR x N(CST; class mean, within-class SD) (Abraham et al. 2022);
           syn_age / syn_male / syn_right_eye drawn from the paper's per-class Table 1 values
        2. VEGF, MCP-1, ANGPTL4 ~ log-normal with the paper's class means (the significant baseline differences)
        3. Panmacular / SRF volume from the real OCT inputs; IL-6 and MMP-1 from anatomy only
           (no significant baseline class difference reported), so they carry no direct class signal.

    cst_mean / cst_sd: how CST is z-scored before being mapped to the paper's scale. None (default) = the input
    cohort's mean / population SD, so the class mix follows PRIOR on average whatever the cohort's CST level.
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    cst = df["CST"].values.astype(float)
    irf = df["IRF"].values.astype(float)
    hrf = df["HRF"].values.astype(float)
    ped = df["PED"].values.astype(float)
    srf = df["SRF"].values.astype(float)

    cst_mean = float(np.mean(cst)) if cst_mean is None else cst_mean
    cst_sd = (float(np.std(cst)) or 1.0) if cst_sd is None else cst_sd
    z_cst = (cst - cst_mean) / cst_sd

    # -------------------------------------------------------------------------
    # 1. Responder class from CST (inverse-CDF draw from the per-eye posterior)
    # -------------------------------------------------------------------------
    post = _cst_class_posterior(cst, cst_mean, cst_sd)
    u = rng.random(n)
    responder_class = (u[:, None] > np.cumsum(post, axis=1)).sum(axis=1).clip(0, 3)

    # Demographics from class (replace the real age / gender / eye as model features)
    age = np.clip(rng.normal(AGE_CLASS_MEAN[responder_class], AGE_CLASS_SD[responder_class]), 18.0, 100.0)
    male = (rng.random(n) < MALE_CLASS_P[responder_class]).astype(int)
    right_eye = (rng.random(n) < RIGHT_EYE_CLASS_P[responder_class]).astype(int)

    # -------------------------------------------------------------------------
    # 2. Anatomical volumes (from the real OCT inputs)
    # -------------------------------------------------------------------------
    pan_vol = 9.8 + 1.2 * z_cst + 0.8 * irf + rng.normal(0.0, 0.45, size=n)
    z_pan = (pan_vol - 9.8) / 1.6

    # SRF volume, hurdle: 0 without SRF, else log-normal conditioned on CST and PED
    active_srf_log = -1.2 + 0.6 * z_cst + 0.5 * ped + rng.normal(0.0, 0.35, size=n)
    srf_vol = np.where(srf == 1.0, np.exp(active_srf_log), 0.0)
    z_srf = np.where(srf == 1.0, (active_srf_log - (-1.2)) / 0.60, 0.0)

    # -------------------------------------------------------------------------
    # 3. Class-dependent cytokines: log-normal with arithmetic mean = paper class mean
    #    (mu_ln = ln(mean) - sigma^2 / 2), floored at the assay LLOQ
    # -------------------------------------------------------------------------
    cyto = {}
    for name, means in CYTOKINE_CLASS_MEAN.items():
        s = CYTOKINE_LOG_SD[name]
        mu = np.log(means[responder_class]) - 0.5 * s ** 2
        cyto[name] = np.maximum(CYTOKINE_LLOQ[name], np.exp(mu + s * rng.normal(0.0, 1.0, size=n)))

    # -------------------------------------------------------------------------
    # 4. Class-independent cytokines (anatomy-driven, as before)
    #    ln(Y) = mu + (r * sigma) * z_X + sigma * sqrt(1 - r^2) * eps
    # -------------------------------------------------------------------------
    r, s = 0.47, 0.95   # IL-6 ~ panmacular volume, +0.35 for HRF (activated microglia)
    il6 = np.maximum(1.5, np.exp(2.1 + r * s * z_pan + 0.35 * hrf + s * np.sqrt(1 - r ** 2) * rng.normal(size=n)))
    r, s = 0.50, 0.75   # MMP-1 ~ SRF volume, +0.40 for PED
    mmp1 = np.maximum(10.0, np.exp(3.2 + r * s * z_srf + 0.40 * ped + s * np.sqrt(1 - r ** 2) * rng.normal(size=n)))

    # -------------------------------------------------------------------------
    # 5. Attach synthetic columns
    # -------------------------------------------------------------------------
    out = df.copy()
    out["syn_age"] = np.round(age, 1)
    out["syn_male"] = male
    out["syn_right_eye"] = right_eye
    out["syn_panmacular_vol_mm3"] = np.round(pan_vol, 3)
    out["syn_srf_vol_mm3"] = np.round(srf_vol, 4)
    out["syn_VEGF_pg_ml"] = np.round(cyto["VEGF"], 2)
    out["syn_IL6_pg_ml"] = np.round(il6, 2)
    out["syn_MCP1_pg_ml"] = np.round(cyto["MCP1"], 2)
    out["syn_MMP1_pg_ml"] = np.round(mmp1, 2)
    out["syn_ANGPTL4_pg_ml"] = np.round(cyto["ANGPTL4"], 2)
    out["syn_responder_class"] = responder_class
    out["syn_responder_label"] = [CLASS_NAMES[c] for c in responder_class]
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 2000
    mock = pd.DataFrame({"CST": rng.normal(415, 175, n).clip(150, 1200),
                         "IRF": rng.random(n) < 0.78, "HRF": rng.random(n) < 0.5,
                         "PED": rng.random(n) < 0.2, "SRF": rng.random(n) < 0.3}).astype(float)
    c = synthesize_oct_proteomics_cohort(mock, seed=101)
    print("class share (target", np.round(PRIOR, 3), "):",
          np.round(np.bincount(c.syn_responder_class, minlength=4) / n, 3))
    cols = ["CST", "syn_age", "syn_male", "syn_right_eye", "syn_VEGF_pg_ml", "syn_MCP1_pg_ml", "syn_ANGPTL4_pg_ml", "syn_IL6_pg_ml"]
    print(c.groupby("syn_responder_label")[cols].mean().round(1).to_string())
