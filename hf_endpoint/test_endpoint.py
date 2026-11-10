"""Check that the endpoint reproduces a run's own test predictions.

Takes test-split eyes from <repo>/<run>/predictions_test.csv, sends their images (from data/) to the
handler, and compares the answers with the predictions saved during training.

    # 1. locally: downloads the run, imports handler.py, no endpoint needed
    python hf_endpoint/test_endpoint.py --run retfound_mae_pre-features_hard-label_dx3_ODOS_pre_idxall_lp

    # 2. the deployed endpoint (URL from the endpoint page; token = your HF token by default)
    python hf_endpoint/test_endpoint.py --run <run> --url https://xxxx.endpoints.huggingface.cloud
"""
import argparse
import base64
import json
import os
import sys
import time

import pandas as pd
import requests
from huggingface_hub import get_token, hf_hub_download, snapshot_download

p = argparse.ArgumentParser()
p.add_argument("--repo", default="angisaac/retfound-oct")
p.add_argument("--run", required=True, help="run folder in the repo")
p.add_argument("--url", default=None, help="endpoint URL; omit to run the handler locally")
p.add_argument("--token", default=None, help="HF token for the endpoint (default: your logged-in token)")
p.add_argument("--data", default="data", help="sample store from prepare_data.py")
p.add_argument("--n", type=int, default=3, help="how many test rows (eyes for MIL runs) to send")
p.add_argument("--tol", type=float, default=0.01, help="max allowed difference (probability, or real value / train std)")
a = p.parse_args()

info = json.load(open(hf_hub_download(a.repo, f"{a.run}/filter.json"), encoding="utf-8"))
expected = pd.read_csv(hf_hub_download(a.repo, f"{a.run}/predictions_test.csv")).head(a.n)
mil = bool((info.get("mil") or {}).get("enabled"))
std = {c: v["std"] for c, v in info.get("target_normalisation", {}).items()}


def slice_paths(row):
    """Image files of one predictions_test.csv row (a whole eye under MIL, else one slice)."""
    if not mil:
        return [os.path.join(a.data, row.sample_id, "image.jpg")]
    return [os.path.join(a.data, f"{row.sample_id}_{int(i):03d}", "image.jpg") for i in str(row.image_idx).split("-")]


if a.url:
    token = a.token or get_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def call(payload):
        r = requests.post(a.url, headers=headers, json=payload, timeout=300)
        if r.status_code != 200:
            sys.exit(f"HTTP {r.status_code}: {r.text[:500]}")
        return r.json()
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from handler import EndpointHandler
    print(f"Downloading {a.repo}/{a.run} and loading it locally ...")
    root = snapshot_download(a.repo, allow_patterns=[f"{a.run}/*"])
    handler = EndpointHandler(root)
    call = handler

worst, failed = 0.0, False
for row in expected.itertuples(index=False):
    paths = slice_paths(row)
    payload = {"inputs": [base64.b64encode(open(f, "rb").read()).decode() for f in paths],
               "parameters": {"run": a.run, "aggregate": "mean" if mil else "none"}}
    t = time.time()
    out = call(payload)
    dt = time.time() - t
    if isinstance(out, dict) and "error" in out:
        sys.exit(f"Endpoint error: {out}")
    pred = (out if mil else out[0])["predictions"]

    print(f"\n{row.sample_id}  ({len(paths)} image(s), {dt:.1f}s)")
    for t_ in info["targets"]:
        c, kind = t_["name"], t_["kind"]
        key = c.replace(" ", "_")
        got = pred[c]["value"] if kind == "real" else pred[c]["probability"]
        want = getattr(row, f"prob_{key}", None) if kind == "binary" else None
        want = getattr(row, f"pred_{key}") if want is None else want
        diff = abs(got - want) / (std.get(c, 1.0) if kind == "real" else 1.0)
        worst = max(worst, diff)
        ok = diff <= a.tol
        failed |= not ok
        extra = f"  label {pred[c]['label']}" if "label" in pred[c] else ""
        print(f"  {c:20s} endpoint {got:10.4f}   training {want:10.4f}   {'OK' if ok else 'MISMATCH'}{extra}")

print(f"\n{'PASS' if not failed else 'FAIL'}: largest difference {worst:.4f} (tolerance {a.tol})")
sys.exit(1 if failed else 0)
