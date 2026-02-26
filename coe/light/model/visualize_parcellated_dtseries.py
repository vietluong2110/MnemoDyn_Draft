#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def load_dtseries(path: str) -> np.ndarray:
    img = nib.load(path)
    data = img.get_fdata()

    # Expected shape is timepoints x parcels for parcellated dtseries.
    # If the first axis is much larger, transpose.
    if data.ndim != 2:
        raise ValueError(f"Expected 2D dtseries data, got shape {data.shape}")
    if data.shape[0] > data.shape[1] and data.shape[0] > 10000:
        data = data.T

    return data


def plot_summary(data: np.ndarray, tr: float, out_dir: str, prefix: str) -> None:
    n_timepoints, n_parcels = data.shape
    t = np.arange(n_timepoints) * tr

    # 1) Global mean time series
    global_mean = data.mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, global_mean, linewidth=1.0)
    ax.set_title("Global Mean Signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean BOLD")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_global_mean.png"), dpi=180)
    plt.close(fig)

    # 2) Parcel x time heatmap (z-scored per parcel)
    z = (data - data.mean(axis=0, keepdims=True)) / (data.std(axis=0, keepdims=True) + 1e-8)
    vmax = np.percentile(np.abs(z), 99)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        z.T,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        extent=[t[0], t[-1], 0, n_parcels],
    )
    ax.set_title("Parcellated Time Series Heatmap (z-score per parcel)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Parcel index")
    fig.colorbar(im, ax=ax, label="z-score")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_heatmap.png"), dpi=180)
    plt.close(fig)

    # 3) Parcel standard deviation distribution
    parcel_std = data.std(axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(parcel_std, bins=40, alpha=0.9)
    ax.set_title("Distribution of Parcel Temporal Std")
    ax.set_xlabel("Std (over time)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_parcel_std_hist.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize parcellated CIFTI dtseries")
    parser.add_argument(
        "--input",
        type=str,
        default="/nas/vhluong/ds005747-download/dtseries/sub-011/sub-011_task-rest_space-MNI305_preproc.dtseries_Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3.dlabel_parcellated.dtseries.nii",
        help="Path to parcellated dtseries file",
    )
    parser.add_argument("--tr", type=float, default=0.72, help="Repetition time in seconds")
    parser.add_argument("--out-dir", type=str, default="./viz_outputs", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = load_dtseries(args.input)
    base = os.path.basename(args.input).replace(".dtseries.nii", "")

    print(f"Loaded: {args.input}")
    print(f"Shape: {data.shape} (timepoints x parcels)")
    print(f"Saving plots to: {os.path.abspath(args.out_dir)}")

    plot_summary(data, args.tr, args.out_dir, base)
    print("Done.")
