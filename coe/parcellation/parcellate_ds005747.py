#!/usr/bin/env python3

import os
import time
from glob import glob

import nibabel as nib
import numpy as np

# Configuration
INPUT_DIR = "/nas/vhluong/ds005747-download/dtseries"
ATLAS_FILE = "/nas/vhluong/brain_atlas/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3.dlabel.nii"
INPUT_PATTERN = "sub-*/sub-*_task-rest_space-MNI305_preproc.dtseries.nii"


def load_cifti_data(file_path):
    """Load CIFTI data and ensure output shape is timepoints x grayordinates."""
    try:
        cifti_img = nib.load(file_path)
        data = cifti_img.get_fdata()
        if data.shape[0] == 91282:
            data = data.T
            print(f"Transposed data from {file_path} to shape {data.shape}")
        return data, cifti_img.header
    except Exception as exc:
        print(f"Error loading {file_path}: {exc}")
        return None, None


def parcellate_timeseries(timeseries_data, atlas_data):
    """Average grayordinates within each non-zero atlas parcel."""
    if timeseries_data.shape[1] != atlas_data.shape[1]:
        raise ValueError(
            f"Timeseries ({timeseries_data.shape[1]}) and atlas ({atlas_data.shape[1]}) "
            "have incompatible spatial dimensions"
        )

    parcel_labels = np.unique(atlas_data[0])
    parcel_labels = parcel_labels[parcel_labels != 0]
    print(f"Found {len(parcel_labels)} parcels in atlas")

    n_timepoints = timeseries_data.shape[0]
    n_parcels = len(parcel_labels)
    parcellated_data = np.zeros((n_timepoints, n_parcels))

    for i, label in enumerate(parcel_labels):
        parcel_mask = atlas_data[0] == label
        parcellated_data[:, i] = np.mean(timeseries_data[:, parcel_mask], axis=1)

    return parcellated_data, parcel_labels


def create_parcellated_header(n_timepoints, n_parcels):
    """Create CIFTI header for parcellated time-series."""
    time_axis = nib.cifti2.SeriesAxis(start=0, step=0.72, size=n_timepoints)
    parcel_axis = nib.cifti2.ScalarAxis([f"Parcel_{i + 1}" for i in range(n_parcels)])
    return nib.cifti2.Cifti2Header.from_axes([time_axis, parcel_axis])


def save_parcellated_data(data, output_path):
    """Save parcellated time-series to CIFTI file."""
    n_timepoints, n_parcels = data.shape
    header = create_parcellated_header(n_timepoints, n_parcels)
    cifti = nib.Cifti2Image(data, header=header)
    cifti.to_filename(output_path)
    print(f"Saved parcellated data to {output_path}")


def process_file(input_path, atlas_data, atlas_name):
    """Parcellate one dtseries file."""
    base, _ = os.path.splitext(input_path)
    output_path = f"{base}_{atlas_name}_parcellated.dtseries.nii"

    if os.path.exists(output_path):
        print(f"Skipping existing output: {output_path}")
        return

    ts_data, _ = load_cifti_data(input_path)
    if ts_data is None:
        return

    try:
        parc_data, _ = parcellate_timeseries(ts_data, atlas_data)
        save_parcellated_data(parc_data, output_path)
    except Exception as exc:
        print(f"Error processing {input_path}: {exc}")


def main():
    atlas_data, _ = load_cifti_data(ATLAS_FILE)
    if atlas_data is None:
        raise RuntimeError(f"Failed to load atlas: {ATLAS_FILE}")

    atlas_name = os.path.splitext(os.path.basename(ATLAS_FILE))[0]
    input_glob = os.path.join(INPUT_DIR, INPUT_PATTERN)
    input_files = sorted(glob(input_glob))

    print(f"Found {len(input_files)} dtseries files in {INPUT_DIR}")

    for index, input_path in enumerate(input_files, start=1):
        print(f"\nProcessing {input_path} ({index}/{len(input_files)})")
        start_time = time.time()
        process_file(input_path, atlas_data, atlas_name)
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as exc:
        print(f"An error occurred: {exc}")
