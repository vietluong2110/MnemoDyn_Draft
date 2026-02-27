#!/usr/bin/env python3
import argparse
import os

import nibabel as nib
import numpy as np


def load_cifti_data(file_path):
    """Load CIFTI data and ensure output shape is timepoints x grayordinates."""
    try:
        cifti_img = nib.load(file_path)
        data = cifti_img.get_fdata()
        if data.shape[0] == 91282:
            data = data.T
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

    n_timepoints = timeseries_data.shape[0]
    n_parcels = len(parcel_labels)
    parcellated_data = np.zeros((n_timepoints, n_parcels))

    for i, label in enumerate(parcel_labels):
        parcel_mask = atlas_data[0] == label
        parcellated_data[:, i] = np.mean(timeseries_data[:, parcel_mask], axis=1)

    return parcellated_data, parcel_labels


def create_parcellated_header(n_timepoints, n_parcels, tr_step):
    """Create CIFTI header for parcellated time-series."""
    time_axis = nib.cifti2.SeriesAxis(start=0, step=tr_step, size=n_timepoints)
    parcel_axis = nib.cifti2.ScalarAxis([f"Parcel_{i + 1}" for i in range(n_parcels)])
    return nib.cifti2.Cifti2Header.from_axes([time_axis, parcel_axis])


def save_parcellated_data(data, output_path, tr_step):
    """Save parcellated time-series to CIFTI file."""
    n_timepoints, n_parcels = data.shape
    header = create_parcellated_header(n_timepoints, n_parcels, tr_step)
    cifti = nib.Cifti2Image(data, header=header)
    cifti.to_filename(output_path)


def process_single_file(input_path, output_path, atlas_data, tr_step):
    """Parcellate one dtseries file."""
    if os.path.exists(output_path):
        return True

    ts_data, ts_header = load_cifti_data(input_path)
    if ts_data is None:
        return False

    # Try to extract TR from the CIFTI header if not provided
    if tr_step is None:
        try:
            series_axis = ts_header.get_axis(0)
            tr_step = series_axis.step
        except Exception:
            print(f"Warning: Could not extract TR from {input_path}, defaulting to 1.0")
            tr_step = 1.0

    try:
        parc_data, _ = parcellate_timeseries(ts_data, atlas_data)
        save_parcellated_data(parc_data, output_path, tr_step)
        return True
    except Exception as exc:
        print(f"Error processing {input_path}: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Parcellate a single CIFTI file.")
    parser.add_argument("--input-cifti", required=True, help="Path to input .dtseries.nii")
    parser.add_argument("--output-cifti", required=True, help="Path to output parcellated cifti")
    parser.add_argument("--atlas-cifti", required=True, help="Path to atlas .dlabel.nii")
    parser.add_argument("--tr", type=float, default=None, help="TR (step) for the time series. Extracted from header if omitted.")
    
    args = parser.parse_args()

    atlas_data, _ = load_cifti_data(args.atlas_cifti)
    if atlas_data is None:
        raise RuntimeError(f"Failed to load atlas: {args.atlas_cifti}")

    process_single_file(args.input_cifti, args.output_cifti, atlas_data, args.tr)


if __name__ == "__main__":
    main()
