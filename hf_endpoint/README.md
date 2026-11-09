# Hugging Face Inference Endpoint

`handler.py` serves runs uploaded by `--hub_repo` (repo layout: `<run>/checkpoint-best.pth` + `<run>/filter.json`).

Deploy:
1. `python hf_endpoint/handler.py --push angisaac/retfound-oct` (puts `handler.py` + `requirements.txt` at the repo root).
2. Create an endpoint on that repo (task "Custom", GPU recommended). Set env `RETFOUND_RUN=<run folder>` if the repo has more than one run.

Local test: `python hf_endpoint/handler.py --path <dir with run folders> [--run NAME] [--aggregate mean] img.jpg ...`

## Request
```json
{"inputs": "<base64 jpg>" | ["<base64>", ...],
 "parameters": {"run": "<run folder>", "aggregate": "mean", "threshold": 0.5}}
```
Raw image bytes (`Content-Type: image/jpeg`) also work. Every parameter is optional.
Images go through the same preprocessing as eval: RGB, bicubic resize, center crop to `input_size`, ImageNet normalisation.

## Response
One object per image (a single object if `inputs` was one image, or if `aggregate="mean"`):
```json
{"run": "...", "task_type": "pre_features", "label_mode": "hard",
 "predictions": {"preCST": {"value": 344.8},
                 "preIRF": {"probability": 0.52, "label": 1}}}
```
- Real targets (CST, VA): `value` in original units (z-score undone with the train mean/std from filter.json).
- Binary targets: `probability` = sigmoid output. `label` (0/1 at `threshold`) only for hard-label runs.
- `aggregate="mean"` averages values/probabilities over all images (e.g. every slice of one eye) and adds `n_images`.
