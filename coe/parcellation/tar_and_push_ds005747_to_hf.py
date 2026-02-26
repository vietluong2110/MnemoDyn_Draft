#!/usr/bin/env python3
"""Create one large tarball of parcellated dtseries files and upload to HF."""

from __future__ import annotations

import argparse
import os
import tarfile
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tar all parcellated ds005747 dtseries files into one archive and upload to Hugging Face."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/nas/vhluong/ds005747-download/dtseries"),
        help="Root directory to scan for parcellated dtseries files.",
    )
    parser.add_argument(
        "--pattern",
        default="*_parcellated.dtseries.nii",
        help="Recursive glob pattern under --input-dir.",
    )
    parser.add_argument(
        "--archive-name",
        default="ds005747_parcellated_all.tar.gz",
        help="Tarball filename to create and upload.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HF dataset repo id, e.g. your-name/ds005747-parcellated",
    )
    parser.add_argument(
        "--remote-subdir",
        default="archives",
        help="Remote folder inside HF repo where archive is uploaded.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create dataset repo as private if missing.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Target branch/revision.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token. If omitted, uses HF_TOKEN env var or cached login.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload ds005747 parcellated tar archive",
        help="Commit message for upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be archived/uploaded without writing or uploading.",
    )
    return parser.parse_args()


def collect_files(input_dir: Path, pattern: str) -> list[Path]:
    files: list[Path] = []
    for p in sorted(input_dir.rglob(pattern)):
        if p.is_file():
            files.append(p)
    return files


def create_tarball(file_paths: list[Path], input_dir: Path, tar_path: Path) -> None:
    with tarfile.open(tar_path, "w:gz") as tar:
        for file_path in file_paths:
            # Keep paths relative to input root.
            arcname = file_path.relative_to(input_dir)
            tar.add(file_path, arcname=str(arcname))


def build_manifest(file_paths: list[Path], input_dir: Path, manifest_path: Path) -> None:
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("relative_path\n")
        for file_path in file_paths:
            f.write(f"{file_path.relative_to(input_dir)}\n")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    token = args.token or os.getenv("HF_TOKEN")
    file_paths = collect_files(input_dir, args.pattern)
    if not file_paths:
        raise RuntimeError(f"No files matched '{args.pattern}' under '{input_dir}'.")

    print(f"Matched files: {len(file_paths)}")
    print(f"Archive: {args.archive_name}")
    print(f"Remote target: https://huggingface.co/datasets/{args.repo_id}/{args.remote_subdir}")

    if args.dry_run:
        for rel in [p.relative_to(input_dir) for p in file_paths[:10]]:
            print(f"[dry-run] {rel}")
        if len(file_paths) > 10:
            print(f"[dry-run] ... ({len(file_paths) - 10} more files)")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it with: pip install huggingface_hub"
        ) from exc

    with tempfile.TemporaryDirectory(prefix="ds005747_tar_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        upload_dir = tmp_path / "upload"
        upload_dir.mkdir(parents=True, exist_ok=True)

        tar_path = upload_dir / args.archive_name
        manifest_path = upload_dir / f"{Path(args.archive_name).stem}.manifest.tsv"

        create_tarball(file_paths, input_dir, tar_path)
        build_manifest(file_paths, input_dir, manifest_path)

        print(f"Tarball created: {tar_path} ({tar_path.stat().st_size / (1024**3):.2f} GiB)")
        print(f"Manifest created: {manifest_path}")

        api = HfApi(token=token)
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(upload_dir),
            path_in_repo=args.remote_subdir.strip("/") or None,
            revision=args.revision,
            commit_message=args.commit_message,
        )

    print(f"Upload complete: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
