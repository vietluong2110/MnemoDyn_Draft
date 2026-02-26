#!/usr/bin/env python3
"""Publish a MnemoDyn Lightning checkpoint to Hugging Face Hub.

This uploads a small, reproducible package:
- model.ckpt
- hparams.yaml (optional)
- metrics.csv (optional)
- README.md (auto-generated model card)
- load_from_hf.py (example loader)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a MnemoDyn checkpoint to Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True, help="Hub repo id, e.g. your-name/mnemodyn-gordon333")
    parser.add_argument(
        "--version-dir",
        type=Path,
        default=None,
        help="Training version directory containing checkpoints/, hparams.yaml, metrics.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a specific .ckpt. If omitted, best checkpoint is selected from --version-dir/checkpoints",
    )
    parser.add_argument("--hparams", type=Path, default=None, help="Optional path to hparams.yaml")
    parser.add_argument("--metrics", type=Path, default=None, help="Optional path to metrics.csv")
    parser.add_argument(
        "--model-class",
        default="coe.light.model.main:LitORionModelOptimized",
        help="Import path used by load_from_hf.py, format module.path:ClassName",
    )
    parser.add_argument("--dataset", default="GordonHCP", help="Dataset name for model card metadata")
    parser.add_argument("--license", default="cc-by-nc-4.0", help="SPDX-like license string for model card")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    parser.add_argument("--revision", default="main", help="Target branch/revision")
    parser.add_argument("--token", default=None, help="HF token. If omitted, uses HF_TOKEN env var or cached login")
    parser.add_argument("--commit-message", default="Add MnemoDyn checkpoint", help="Commit message")
    parser.add_argument(
        "--remote-subdir",
        default=None,
        help=(
            "Remote folder inside the HF repo (e.g. Orion_333). "
            "If omitted and --version-dir is provided, defaults to a sensible name "
            "(parent folder for lightning 'version_*' dirs, otherwise version-dir name)."
        ),
    )
    return parser.parse_args()


def _extract_val_mae(ckpt_name: str) -> float:
    match = re.search(r"val_mae=([-+eE0-9.]+)", ckpt_name)
    if not match:
        return float("inf")
    try:
        return float(match.group(1))
    except ValueError:
        return float("inf")


def find_best_checkpoint(version_dir: Path) -> Path:
    ckpt_dir = version_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    ckpts = sorted(p for p in ckpt_dir.iterdir() if p.suffix == ".ckpt")
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found under: {ckpt_dir}")
    return min(ckpts, key=lambda p: _extract_val_mae(p.name))


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None]:
    ckpt = args.checkpoint
    hparams = args.hparams
    metrics = args.metrics

    if args.version_dir is not None:
        version_dir = args.version_dir
        if ckpt is None:
            ckpt = find_best_checkpoint(version_dir)
        if hparams is None:
            candidate = version_dir / "hparams.yaml"
            hparams = candidate if candidate.exists() else None
        if metrics is None:
            candidate = version_dir / "metrics.csv"
            metrics = candidate if candidate.exists() else None

    if ckpt is None:
        raise ValueError("Provide --checkpoint or --version-dir.")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if hparams is not None and not hparams.exists():
        raise FileNotFoundError(f"hparams file not found: {hparams}")
    if metrics is not None and not metrics.exists():
        raise FileNotFoundError(f"metrics file not found: {metrics}")

    return ckpt, hparams, metrics


def infer_remote_subdir(args: argparse.Namespace) -> str | None:
    if args.remote_subdir:
        return args.remote_subdir.strip("/").strip()
    if args.version_dir is not None:
        version_name = args.version_dir.name
        # Common Lightning layout: .../<experiment>/version_17
        # Prefer the experiment folder name so uploads group by model family.
        if re.fullmatch(r"version_\d+", version_name) and args.version_dir.parent != args.version_dir:
            return args.version_dir.parent.name
        return version_name
    return None


def build_model_card(args: argparse.Namespace, ckpt_name: str, remote_subdir: str | None) -> str:
    ckpt_remote = f"{remote_subdir}/model.ckpt" if remote_subdir else "model.ckpt"
    hparams_remote = f"{remote_subdir}/hparams.yaml" if remote_subdir else "hparams.yaml"
    metrics_remote = f"{remote_subdir}/metrics.csv" if remote_subdir else "metrics.csv"
    loader_remote = f"{remote_subdir}/load_from_hf.py" if remote_subdir else "load_from_hf.py"
    return f"""---
license: {args.license}
library_name: pytorch-lightning
tags:
  - neuroscience
  - fmri
  - time-series
  - pytorch-lightning
---

# MnemoDyn Checkpoint

This repository contains a MnemoDyn checkpoint exported from this codebase.

## Checkpoint

- Source checkpoint: `{ckpt_name}`
- Dataset: `{args.dataset}`
- Model class: `{args.model_class}`
- Remote directory: `{remote_subdir or "(repo root)"}`

## Files

- `{ckpt_remote}`: Lightning checkpoint
- `{hparams_remote}`: training hyperparameters (if available)
- `{metrics_remote}`: training/validation metrics (if available)
- `{loader_remote}`: minimal loading script

## Usage

```bash
python {loader_remote} --repo_id {args.repo_id}
```
"""


def build_loader_script(remote_subdir: str | None) -> str:
    remote_ckpt = f"{remote_subdir}/model.ckpt" if remote_subdir else "model.ckpt"
    return """#!/usr/bin/env python3
import argparse
from huggingface_hub import hf_hub_download
from coe.light.model.main import LitORionModelOptimized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True, help="e.g. your-name/mnemodyn-gordon333")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    ckpt_path = hf_hub_download(repo_id=args.repo_id, filename="%s", revision=args.revision)
    model = LitORionModelOptimized.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    print("Loaded model from", ckpt_path)


if __name__ == "__main__":
    main()
""" % remote_ckpt


def main() -> None:
    args = parse_args()
    ckpt, hparams, metrics = resolve_paths(args)
    remote_subdir = infer_remote_subdir(args)

    token = args.token or os.getenv("HF_TOKEN")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it with: pip install huggingface_hub"
        ) from exc

    with tempfile.TemporaryDirectory(prefix="mnemodyn_hf_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        shutil.copy2(ckpt, tmp_path / "model.ckpt")
        if hparams is not None:
            shutil.copy2(hparams, tmp_path / "hparams.yaml")
        if metrics is not None:
            shutil.copy2(metrics, tmp_path / "metrics.csv")

        (tmp_path / "README.md").write_text(build_model_card(args, ckpt.name, remote_subdir))
        (tmp_path / "load_from_hf.py").write_text(build_loader_script(remote_subdir))

        api = HfApi(token=token)
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(tmp_path),
            path_in_repo=remote_subdir,
            revision=args.revision,
            commit_message=args.commit_message,
        )

    print(f"Uploaded checkpoint package to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
