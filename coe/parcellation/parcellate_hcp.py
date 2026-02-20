#!/usr/bin/env python3

import os
import numpy as np
import nibabel as nib
import time

# Configuration
INPUT_DIR = "/nas/sourav/GordonHCP/HCP_Dense"  # Same as your download directory
ATLAS_FILE = "/home/vhluong/BrainLM/toolkit/atlases/A424.dlabel.nii"  # Path to your atlas file
INPUT_FILENAME = "rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii"

def load_cifti_data(file_path):
	"""Load CIFTI data and ensure correct orientation"""
	try:
		cifti_img = nib.load(file_path)
		data = cifti_img.get_fdata()
		# If rows are 91282 (grayordinates), transpose to 1 x 91282
		if data.shape[0] == 91282:
			data = data.T  # Transpose to 1 x 91282
			print(f"Transposed data from {file_path} to shape {data.shape}")
		return data, cifti_img.header
	except Exception as e:
		print(f"Error loading {file_path}: {str(e)}")
		return None, None

def parcellate_timeseries(timeseries_data, atlas_data):
	"""Parcellate dense timeseries using the atlas (all 352 regions)"""
	if timeseries_data.shape[1] != atlas_data.shape[1]:
		raise ValueError(f"Timeseries ({timeseries_data.shape[1]}) and atlas ({atlas_data.shape[1]}) have incompatible spatial dimensions")
	
	# Get unique parcel labels, excluding 0 (background)
	parcel_labels = np.unique(atlas_data[0])
	parcel_labels = parcel_labels[parcel_labels != 0]  # Use 1-352
	print(f"Found {len(parcel_labels)} parcels in atlas (full 352 regions)")
	
	# Initialize output array: timepoints x parcels
	n_timepoints = timeseries_data.shape[0]
	n_parcels = len(parcel_labels)
	parcellated_data = np.zeros((n_timepoints, n_parcels))
	
	# Compute mean timeseries for each parcel
	for i, label in enumerate(parcel_labels):
		parcel_mask = atlas_data[0] == label
		parcellated_data[:, i] = np.mean(timeseries_data[:, parcel_mask], axis=1)
	
	return parcellated_data, parcel_labels

def create_parcellated_header(n_timepoints, n_parcels):
	"""Create a new CIFTI header for parcellated data"""
	# Time axis (rows)
	time_axis = nib.cifti2.SeriesAxis(start=0, step=0.72, size=n_timepoints)  # Assuming TR=0.72s from HCP
	# Parcel axis (columns)
	parcel_axis = nib.cifti2.ScalarAxis([f"Parcel_{i+1}" for i in range(n_parcels)])
	# Create header from axes
	new_header = nib.cifti2.Cifti2Header.from_axes([time_axis, parcel_axis])
	return new_header

def save_parcellated_data(data, output_path, parcel_labels):
	"""Save parcellated data with a new CIFTI header"""
	n_timepoints, n_parcels = data.shape
	new_header = create_parcellated_header(n_timepoints, n_parcels)
	new_cifti = nib.Cifti2Image(data, header=new_header)
	new_cifti.to_filename(output_path)
	print(f"Saved parcellated data to {output_path}")

def process_subject(subject_id, atlas_file):
	"""Process a single subject's data"""
	subject_dir = os.path.join(INPUT_DIR, subject_id)
	input_path = os.path.join(subject_dir, INPUT_FILENAME)
	
	atlas_name = os.path.splitext(os.path.basename(atlas_file))[0]
	output_filename = f"rfMRI_REST1_LR_Atlas_hp2000_clean_{atlas_name}_parcellated.dtseries.nii"
	output_path = os.path.join(subject_dir, output_filename)
	
	if not os.path.exists(input_path):
		print(f"Skipping {subject_id}: Input file not found")
		return
	
	if os.path.exists(output_path):
		print(f"Skipping {subject_id}: Output file already exists")
		return
	
	# Load data
	timeseries_data, ts_header = load_cifti_data(input_path)
	atlas_data, _ = load_cifti_data(atlas_file)
	
	if timeseries_data is None or atlas_data is None:
		return
	
	# Parcellate
	try:
		parcellated_data, parcel_labels = parcellate_timeseries(timeseries_data, atlas_data)
		save_parcellated_data(parcellated_data, output_path, parcel_labels)
	except Exception as e:
		print(f"Error processing {subject_id}: {str(e)}")

def main():
	subjects = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
	print(f"Found {len(subjects)} subjects")
	
	for i, subject in enumerate(subjects, 1):
		print(f"\nProcessing subject {subject} ({i}/{len(subjects)})")
		start_time = time.time()
		process_subject(subject, ATLAS_FILE)
		elapsed = time.time() - start_time
		print(f"Completed {subject} in {elapsed:.2f} seconds")

if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("\nProcessing interrupted by user")
	except Exception as e:
		print(f"An error occurred: {str(e)}")