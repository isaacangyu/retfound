"""Hugging Face Inference Endpoint handler for RETFound OCT benchmark runs.

The model repo (e.g. angisaac/retfound-oct, filled by `--hub_repo`) holds one folder per run:

    <repo>/handler.py                 <- this file
    <repo>/requirements.txt           <- hf_endpoint/requirements.txt
    <repo>/<run>/checkpoint-best.pth
    <repo>/<run>/filter.json          <- targets, label mode, CST/VA normalisation, input size

Pick the run with the endpoint env var RETFOUND_RUN, or per request with parameters.run.
If neither is set and the repo has exactly one run, that run is used.

Request:  {"inputs": <image | [images]>, "parameters": {...}}
Response: one dict per image (or one dict when aggregate="mean"); see README in this folder.

Local test:   python hf_endpoint/handler.py --path <dir with run folders> img1.jpg img2.jpg
Upload code:  python hf_endpoint/handler.py --push angisaac/retfound-oct
"""
import base64
import io
import json
import os
from functools import partial

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Model (same architecture as models_vit.py / main_finetune.py)
# ---------------------------------------------------------------------------

def _retfound_mae(img_size, num_classes, global_pool):
    import timm.models.vision_transformer as tvt

    class VisionTransformer(tvt.VisionTransformer):
        def __init__(self, global_pool=False, **kw):
            super().__init__(**kw)
            self.global_pool = global_pool
            if global_pool:
                self.fc_norm = kw["norm_layer"](kw["embed_dim"])
                del self.norm

        def forward_features(self, x):
            x = self.patch_embed(x)
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = self.pos_drop(x + self.pos_embed)
            for blk in self.blocks:
                x = blk(x)
            if self.global_pool:
                return self.fc_norm(x[:, 1:, :].mean(dim=1, keepdim=True))
            return self.norm(x)[:, 0]

    return VisionTransformer(
        global_pool=global_pool, img_size=img_size, num_classes=num_classes,
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))


DINOV2_ARCH = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2_vitg14": "vit_giant_patch14_dinov2.lvd142m",
}


def _build_model(name, model_arch, input_size, state_dict):
    """Rebuild the network; the head shape (linear or MLP) is read from the checkpoint."""
    if "head.0.weight" in state_dict:  # MLP head: Linear-GELU-Dropout-Linear
        hidden, in_features = state_dict["head.0.weight"].shape
        n_out = state_dict["head.3.weight"].shape[0]
    else:
        n_out = state_dict["head.weight"].shape[0]
        hidden = None

    if name == "RETFound_mae":
        model = _retfound_mae(input_size, n_out, global_pool="fc_norm.weight" in state_dict)
    elif name in ("RETFound_dinov2", "Dinov2"):
        import timm
        arch = DINOV2_ARCH["dinov2_vitl14" if name == "RETFound_dinov2" else model_arch]
        model = timm.create_model(arch, pretrained=False, img_size=224, num_classes=n_out)
    elif name == "Dinov3":
        model = torch.hub.load("facebookresearch/dinov3", model_arch, pretrained=False, trust_repo=True)
        feat = getattr(model, "embed_dim", None) or getattr(model, "num_features")
        model.head = nn.Linear(feat, n_out)
    else:
        raise ValueError(f"Unsupported model '{name}'")

    if hidden is not None:
        model.head = nn.Sequential(nn.Linear(in_features, hidden), nn.GELU(), nn.Dropout(0.0),
                                   nn.Linear(hidden, n_out))
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


# ---------------------------------------------------------------------------
# One trained run
# ---------------------------------------------------------------------------

class Run:
    def __init__(self, run_dir, device):
        with open(os.path.join(run_dir, "filter.json"), encoding="utf-8") as f:
            self.info = json.load(f)
        # weights_only=False: the checkpoint stores the training argparse Namespace (needs pandas).
        ckpt = torch.load(os.path.join(run_dir, "checkpoint-best.pth"), map_location="cpu", weights_only=False)
        args = ckpt.get("args")
        name = getattr(args, "model", None) or self.info["model"]
        input_size = int(getattr(args, "input_size", None) or self.info.get("input_size") or 224)
        model_arch = getattr(args, "model_arch", None)

        self.name = self.info["run"]
        self.targets = [(t["name"], t["kind"]) for t in self.info["targets"]]
        self.label_mode = self.info.get("label_mode", "soft")
        self.norm = self.info.get("target_normalisation", {})
        self.epoch = ckpt.get("epoch")
        self.device = device
        self.model = _build_model(name, model_arch, input_size, ckpt["model"]).to(device)
        del ckpt

        # Same eval transform as util/datasets.py
        crop_pct = 224 / 256 if input_size <= 224 else 1.0
        self.transform = transforms.Compose([
            transforms.Resize(int(input_size / crop_pct), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    @torch.no_grad()
    def predict(self, images, batch_size=16):
        """PIL images -> raw outputs [N, n_targets] (logits for binary, z-scores for real)."""
        outs = []
        for i in range(0, len(images), batch_size):
            x = torch.stack([self.transform(im) for im in images[i:i + batch_size]]).to(self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                outs.append(self.model(x).float().cpu())
        return torch.cat(outs)

    def decode(self, raw):
        """Raw outputs [N, T] -> per-target values in natural units [N, T]."""
        vals = raw.clone()
        for j, (c, kind) in enumerate(self.targets):
            if kind == "binary":
                vals[:, j] = torch.sigmoid(raw[:, j])
            else:
                vals[:, j] = raw[:, j] * self.norm[c]["std"] + self.norm[c]["mean"]
        return vals

    def format(self, row, threshold):
        out = {}
        for j, (c, kind) in enumerate(self.targets):
            v = float(row[j])
            if kind == "real":
                out[c] = {"value": v}
            elif self.label_mode == "hard":
                out[c] = {"probability": v, "label": int(v >= threshold)}
            else:
                out[c] = {"probability": v}
        return out


# ---------------------------------------------------------------------------
# Endpoint entry point
# ---------------------------------------------------------------------------

def _to_image(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x)).convert("RGB")
    if isinstance(x, str):
        if x.startswith("data:"):
            x = x.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(x))).convert("RGB")
    if isinstance(x, dict) and "image" in x:
        return _to_image(x["image"])
    raise ValueError(f"Unsupported input type {type(x).__name__}; send a base64 string or raw image bytes")


class EndpointHandler:
    def __init__(self, path=""):
        self.path = path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.runs = {}
        self.available = sorted(d for d in os.listdir(path)
                                if os.path.isfile(os.path.join(path, d, "checkpoint-best.pth"))
                                and os.path.isfile(os.path.join(path, d, "filter.json")))
        self.default = os.environ.get("RETFOUND_RUN") or (self.available[0] if len(self.available) == 1 else None)
        if self.default:
            self._run(self.default)

    def _run(self, name):
        if name not in self.runs:
            if name not in self.available:
                raise ValueError(f"Unknown run '{name}'. Available: {self.available}")
            self.runs[name] = Run(os.path.join(self.path, name), self.device)
        return self.runs[name]

    def __call__(self, data):
        params = data.get("parameters") or {}
        name = params.get("run") or self.default
        if not name:
            return {"error": "Set parameters.run (or env RETFOUND_RUN)", "available_runs": self.available}
        run = self._run(name)

        inputs = data.get("inputs", data)
        single = not isinstance(inputs, list)
        images = [_to_image(x) for x in ([inputs] if single else inputs)]
        if not images:
            return {"error": "no images in 'inputs'"}
        threshold = float(params.get("threshold", 0.5))

        vals = run.decode(run.predict(images))
        meta = {"run": run.name, "task_type": run.info["task_type"], "label_mode": run.label_mode}
        if params.get("aggregate") == "mean":  # e.g. all slices of one eye -> one prediction
            return {**meta, "n_images": len(images), "predictions": run.format(vals.mean(0), threshold)}
        preds = [{**meta, "predictions": run.format(r, threshold)} for r in vals]
        return preds[0] if single else preds


# ---------------------------------------------------------------------------
# CLI: local test / upload handler to the hub repo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="*", help="image files to run through the handler")
    p.add_argument("--path", default=".", help="folder containing <run>/checkpoint-best.pth + filter.json")
    p.add_argument("--run", default=None)
    p.add_argument("--aggregate", default=None, choices=[None, "mean"])
    p.add_argument("--push", metavar="REPO", help="upload handler.py + requirements.txt to this hub repo")
    a = p.parse_args()

    if a.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.push, private=True, exist_ok=True)  # same as --hub_repo: private model repo
        here = os.path.dirname(os.path.abspath(__file__))
        for f in ("handler.py", "requirements.txt", "README.md"):
            api.upload_file(path_or_fileobj=os.path.join(here, f), path_in_repo=f, repo_id=a.push,
                                commit_message=f"endpoint: {f}")
            print(f"uploaded {f} -> {a.push}/{f}")
    if a.images:
        h = EndpointHandler(a.path)
        payload = [base64.b64encode(open(f, "rb").read()).decode() for f in a.images]
        params = {k: v for k, v in (("run", a.run), ("aggregate", a.aggregate)) if v}
        print(json.dumps(h({"inputs": payload, "parameters": params}), indent=2))
