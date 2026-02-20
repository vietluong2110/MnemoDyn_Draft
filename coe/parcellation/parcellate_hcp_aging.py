#!/usr/bin/env python3

import os
import numpy as np
import nibabel as nib
import time

# Configuration
INPUT_DIR = "/nas/sourav/HCPAging/dtseries"  # Root directory containing subject folders\N
ATLAS_FILE = "/nas/vhluong/brain_atlas/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3.dlabel.nii"  # Path to your atlas file
SUBFOLDER = "unprocessed/rfMRI_REST2_AP"  # Relative path under each subject
dtseries_suffix = "_rfMRI_REST2_AP.dtseries.nii"  # Suffix for input filenames

def load_cifti_data(file_path):
    """Load CIFTI data and ensure correct orientation"""
    try:
        cifti_img = nib.load(file_path)
        data = cifti_img.get_fdata()
        # If rows are grayordinates (91282), transpose to timepoints x grayordinates
        if data.shape[0] == 91282:
            data = data.T  # Timepoints x 91282
            print(f"Transposed data from {file_path} to shape {data.shape}")
        return data, cifti_img.header
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def parcellate_timeseries(timeseries_data, atlas_data):
    """Parcellate dense timeseries using the atlas (all 352 regions)"""
    if timeseries_data.shape[1] != atlas_data.shape[1]:
        raise ValueError(
            f"Timeseries spatial dim {timeseries_data.shape[1]} != atlas dim {atlas_data.shape[1]}"
        )
    parcel_labels = np.unique(atlas_data[0])
    parcel_labels = parcel_labels[parcel_labels != 0]
    print(f"Found {len(parcel_labels)} parcels in atlas")
    n_timepoints = timeseries_data.shape[0]
    n_parcels = len(parcel_labels)
    parcellated_data = np.zeros((n_timepoints, n_parcels))
    for i, label in enumerate(parcel_labels):
        mask = atlas_data[0] == label
        parcellated_data[:, i] = np.mean(timeseries_data[:, mask], axis=1)
    return parcellated_data, parcel_labels

def create_parcellated_header(n_timepoints, n_parcels):
    """Create a new CIFTI header for parcellated data"""
    time_axis = nib.cifti2.SeriesAxis(start=0, step=0.72, size=n_timepoints)
    parcel_axis = nib.cifti2.ScalarAxis([f"Parcel_{i+1}" for i in range(n_parcels)])
    return nib.cifti2.Cifti2Header.from_axes([time_axis, parcel_axis])

def save_parcellated_data(data, output_path, parcel_labels):
    """Save parcellated data with a new CIFTI header"""
    n_timepoints, n_parcels = data.shape
    header = create_parcellated_header(n_timepoints, n_parcels)
    img = nib.Cifti2Image(data, header=header)
    img.to_filename(output_path)
    print(f"Saved parcellated data to {output_path}")

def process_subject(subject_id, atlas_file):
    """Process a single subject's data"""
    subject_dir = os.path.join(INPUT_DIR, subject_id, SUBFOLDER)
    input_filename = subject_id + dtseries_suffix
    input_path = os.path.join(subject_dir, input_filename)
    output_filename = subject_id + dtseries_suffix.replace(
        ".dtseries", f"450_parcellated.dtseries"
    )
    output_path = os.path.join(subject_dir, output_filename)

    if not os.path.exists(input_path):
        print(f"Skipping {subject_id}: input not found at {input_path}")
        return
    if os.path.exists(output_path):
        print(output_path)
        print(f"Skipping {subject_id}: already parcellated")
        return

    ts_data, _ = load_cifti_data(input_path)
    atlas_data, _ = load_cifti_data(atlas_file)
    if ts_data is None or atlas_data is None:
        return
    try:
        parc_data, labels = parcellate_timeseries(ts_data, atlas_data)
        save_parcellated_data(parc_data, output_path, labels)
    except Exception as e:
        print(f"Error processing {subject_id}: {e}")

def main():
    subjects = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d)) and not d.startswith('_to_delete')]
    print(f"Found {len(subjects)} subjects in {INPUT_DIR}")
    for i, subj in enumerate(subjects, 1):
        print(f"\nProcessing {subj} ({i}/{len(subjects)})")
        start = time.time()
        process_subject(subj, ATLAS_FILE)
        print(f"Completed {subj} in {time.time() - start:.2f}s")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
