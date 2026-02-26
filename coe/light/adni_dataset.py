import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchcde
import pandas as pd
import nibabel as nib   # for loading .dtseries.nii
from tqdm import tqdm

import os
import pandas as pd
from typing import Tuple, List, Dict
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchcde
import nibabel as nib
from tqdm import tqdm
from typing import List, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

import pdb

class ADNI_Dataset(Dataset):
	def __init__(
		self,
		data_dir: str,
		file_list: List[str],
		label_map: Dict[str, Tuple[str, float]],
		time_step: float,
		interpol: str,
		one_channel: int = None,
		subset: bool = False,
		dim_D: List[int] = None,
		target_length: int = 1200,
		normalizer = None, 
		num_parcels: int = 333,
		label_column = 'subjectAge'
	):
		"""
		data_dir:        root directory, e.g. "/nas/sourav/fresh_ADNI/dtseries/"
		file_list:       list of relative paths (under data_dir) to '*.dtseries.nii'
		label_map:       dict mapping each relative path → (researchGroup, subjectAge)
						 (obtained via load_adni_file_list).
		time_step:       passed through __getitem__
		interpol:        'linear' or 'spline'
		one_channel, subset, dim_D: same as before
		target_length:   target number of timepoints
		num_parcels:     expected number of parcels in that dtseries
		"""
		self.data_dir   = data_dir
		self.file_list  = file_list
		self.label_map  = label_map
		self.time_step  = time_step
		self.interpol   = interpol
		self.one_channel= one_channel
		self.subset     = subset
		self.dim_D      = dim_D
		self.target_length = target_length
		self.normalizer = normalizer
		self.num_parcels   = num_parcels
		self.label_column = label_column

		# 1) Preload data arrays in a list
		raw_data = []
		labels   = []

		print(f"⏳ Preloading {len(self.file_list)} dtseries files from '{data_dir}'...")
		for rel_path in tqdm(self.file_list, desc="Preloading dtseries", unit="file"):
			full_path = os.path.join(data_dir, rel_path)
			if not os.path.isfile(full_path):
				raise FileNotFoundError(f"Expected '{full_path}' not found.")

			# ─── Load the .dtseries.nii via nibabel ───────────────────────────
			img = nib.load(full_path)
			arr: np.ndarray = img.get_fdata().astype(np.float32)  # shape [T, D]
			if arr.ndim != 2:
				raise RuntimeError(f"'{rel_path}' did not load as a 2D array. Got shape {arr.shape}.")
			T, D = arr.shape
			if D != num_parcels:
				raise RuntimeError(
					f"'{rel_path}' has D={D} parcels, but expected num_parcels={num_parcels}."
				)
			x_tensor = torch.from_numpy(arr)  # shape [T, D]
			# print(x_tensor.shape)
			# ─── Channel/subset selection ────────────────────────────────────
			if self.one_channel >= 0:
				x_tensor = x_tensor[:, [one_channel]]
			elif subset:
				
				x_tensor = x_tensor[:, :dim_D]
			# print(x_tensor.shape)
			# ─── Pad or trim to target_length ────────────────────────────────
			reps = int(np.ceil(target_length / x_tensor.shape[0]))
			x_tensor = x_tensor.repeat((reps, 1))[:target_length]
			# x_tensor = self.normalizer.transform_single(x_tensor.unsqueeze(0)).squeeze(0)
			raw_data.append(x_tensor)

			# ─── Lookup labels by relative path ─────────────────────────────
			# label_map was built so that
			#   label_map[rel_path] = (researchGroup, subjectAge)
			if rel_path not in self.label_map:
				raise KeyError(f"No label found for '{rel_path}'.")
			group, age = self.label_map[rel_path]
			labels.append((group, age))

		# Stack everything into a single tensor [N, T, D_sel]
		self.data = torch.stack(raw_data)

		# (Optional) Fit a normalizer here if you like:
		# self.normalizer = Normalizer_update(normalize=True)
		# self.normalizer.fit(self.data)
		# self.data = self.normalizer.transform(self.data)

		# ─── Precompute interpolation coefficients ─────────────────────────
		interp_fn = {
			'linear': torchcde.linear_interpolation_coeffs,
			'spline': torchcde.hermite_cubic_coefficients_with_backward_differences,
		}.get(interpol)
		if interp_fn is None:
			raise ValueError(f"Invalid interpol='{interpol}'. Use 'linear' or 'spline'.")

		# If torchcde allows batched input [N, T, D_sel], do:
		self.coeffs = interp_fn(self.data)

		self.labels = labels  # list of (group, age)

	def __getitem__(self, idx: int):
		x      = self.data[idx]       # [target_length, D_sel]
		coeffs = self.coeffs[idx]     # interpolation coefficients
		(grp, age) = self.labels[idx]
		if self.label_column == 'subjectAge':
			return x, coeffs, age, self.time_step
		if self.label_column == 'researchGroup':
			if grp == 1:
				x = x + 0.1 * torch.randn_like(x)
				return x, coeffs, grp, self.time_step
			else:
				return x, coeffs, grp, self.time_step
		elif self.label_column == 'subjectSex':
			return x, coeffs, grp, self.time_step

	def __len__(self) -> int:
		return len(self.data)

class ADNIDataModule:
	def __init__(self,
			   data_dir,
			   file_list, 
			   label_map, 
			   duration, 
			   original_length, 
			   interpol, 
			   target_length, 
			   num_parcels, 
			   one_channel, 
			   subset, 
			   dim_D,
			   normalize,
			   label_column
			   ):
		self.time_step = torch.from_numpy(np.arange(0, duration, duration/original_length)).float()
		self.label_map = label_map
		self.data_dir = data_dir
		self.interpol_method = interpol
		self.target_length = target_length
		self.num_parcels = num_parcels
		self.one_channel = one_channel
		self.normalize = normalize
		self.subset = subset
		self.dim_D = dim_D
		self.label_column = label_column
		if label_column == 'researchGroup' or 'subjectSex':
			labels = [label_map[f][0] for f in file_list]
			self.train_files, self.test_files, train_labels, test_labels = train_test_split(file_list, labels, test_size=0.2, random_state=42)
		else:
			self.train_files, self.test_files = train_test_split(file_list, test_size=0.2, random_state=42)
		print("Train/Test split: %d/%d", len(self.train_files), len(self.test_files))

	def compute_normalization_stats(self):
		n_total = 0
		sum_, sum_sq = None, None
		gmin, gmax = float('inf'), float('-inf')
		pvmin, pvmax = None, None
		all_samples = []

		for fn in tqdm(self.train_files, desc='Normalization stats'):
			full_path = os.path.join(self.data_dir, fn)
			if not os.path.isfile(full_path):
				raise FileNotFoundError(f"Expected '{full_path}' not found.")

			# ─── Load the .dtseries.nii via nibabel ───────────────────────────
			img = nib.load(full_path)
			data = img.get_fdata().astype(np.float32)  # shape [T, D]

			reps = int(np.ceil(self.target_length / data.shape[0]))
			data = np.tile(data, (reps, 1))[:self.target_length]

			if self.one_channel >= 0:
				data = np.expand_dims(data[:, self.one_channel], -1)
			else:
				if self.subset:
					data = data[:, :self.dim_D]
			t = torch.tensor(data, dtype=torch.float32)
			all_samples.append(t)

			n_total += t.shape[0]
			sum_ = sum_ + t.sum(0) if sum_ is not None else t.sum(0)
			sum_sq = sum_sq + (t ** 2).sum(0) if sum_sq is not None else (t ** 2).sum(0)
			pvmin = torch.min(pvmin, t.min(0).values) if pvmin is not None else t.min(0).values
			pvmax = torch.max(pvmax, t.max(0).values) if pvmax is not None else t.max(0).values
			gmin, gmax = min(gmin, t.min().item()), max(gmax, t.max().item())

		mean = sum_ / n_total
		std = (sum_sq / n_total - mean ** 2).sqrt()

		# Stack data for median, iqr, 99th percentile (in memory)
		data_tensor = torch.stack(all_samples)  # shape: [N, 1200, D]
		del all_samples  # free list memory

		X_flat = data_tensor.view(-1, data_tensor.shape[-1]).numpy()  # shape: [N*1200, D]
		del data_tensor  # free large tensor

		global_median = np.median(X_flat, axis=0)
		global_99th = np.percentile(X_flat, 99, axis=0)
		q25, q75 = np.percentile(X_flat, [25, 75], axis=0)
		global_iqr = q75 - q25
		del X_flat
		return {
			'global_min': gmin,
			'global_max': gmax,
			'global_mean': mean,
			'global_std': std,
			'per_voxel_min': pvmin,
			'per_voxel_max': pvmax,
			'global_median': global_median.tolist(),
			'global_99th': global_99th.tolist(),
			'global_iqr': global_iqr.tolist()
		}


	def setup(self):
		stats = self.compute_normalization_stats()
		# Optional: Pretty-print normalization stats
		print(f"Normalization stats for method '{self.normalize}':")
		

		# Sanity check for required keys
		required_keys = {
			'robust_scaler': ['global_median', 'global_iqr'],
			'subtract_mean_global_std': ['global_std'],
			'subtract_mean_99th_percentile': ['global_99th'],
			'per_voxel_all_patient': ['per_voxel_min', 'per_voxel_max'],
			'all_patient_all_voxel': ['global_min', 'global_max'],
		}
		for key, required in required_keys.items():
			if self.normalize == key:
				for rk in required:
					if rk not in stats:
						raise ValueError(f"Missing key '{rk}' in computed stats for normalization method '{key}'")


		self.train_dataset = ADNI_Dataset(self.data_dir, self.train_files, self.label_map, self.time_step, 
									self.interpol_method, self.one_channel, self.subset, self.dim_D, 
									self.target_length, self.num_parcels, self.label_column)
		self.test_dataset = ADNI_Dataset(self.data_dir, self.test_files, self.label_map, self.time_step, 
									self.interpol_method, self.one_channel, self.subset, self.dim_D, 
									self.target_length, self.num_parcels, self.label_column)
		

	def train_dataloader(self, train_bs):
		return DataLoader(self.train_dataset, batch_size=train_bs, shuffle=True)

	def test_dataloader(self, test_bs):
		return DataLoader(self.test_dataset, batch_size=test_bs, shuffle = False)

def load_adni_file_list(
	data_dir: str,
	metadata_path: str,
	label_column: str,
) -> Tuple[List[str], Dict[str, Tuple[str, float, str]]]:
	"""
	1) Read the metadata CSV which must have columns:
	   relative_path, researchGroup, subjectAge, subjectSex.
	2) For each row, check that the file exists under data_dir/relative_path.
	3) Return the list of valid relative_paths + a map from
	   relative_path -> (researchGroup, subjectAge, subjectSex).
	"""
	# 1) Read & validate metadata
	logger.info(f"Loading metadata from {metadata_path!r}")
	meta = pd.read_csv(metadata_path)
	meta.columns = [c.strip() for c in meta.columns]

	required = {'relative_path', 'researchGroup', 'subjectAge', 'subjectSex'}
	missing = required - set(meta.columns)
	if missing:
		raise ValueError(f"Metadata CSV is missing columns: {missing}")

	# 2) Build a dict of the metadata rows
	label_map_by_relpath: Dict[str, Tuple[str, float, str]] = {}
	for _, row in meta.iterrows():
		rel = row['relative_path'].strip()
		grp = row['researchGroup'].strip()
		age = float(row['subjectAge'])
		sex = row['subjectSex'].strip()

		# (Optional) filter out unwanted groups
		if label_column == 'researchGroup' and grp in {'SMC','Patient','EMCI','LMCI','AD'}:
			continue
		rel = str(rel).replace(
			'_parcellated.dtseries.nii',
			'_450_parcellated.dtseries.nii'
			)
		label_map_by_relpath[rel] = (grp, age, sex)

	logger.info("Metadata entries after filter: %d", len(label_map_by_relpath))

	# 3) Check each metadata path actually exists on disk
	valid_file_list: List[str] = []
	missing_files = []
	for rel, labels in tqdm(label_map_by_relpath.items(), desc="Validating files"):
		full = os.path.join(data_dir, rel)
		if os.path.isfile(full):

			valid_file_list.append(rel)
		else:
			missing_files.append(rel)

	if missing_files:
		logger.warning(
			"The following metadata paths were not found on disk (%d):\n%s",
			len(missing_files),
			"\n".join(missing_files[:5]) + ("...\n" if len(missing_files)>5 else "")
		)

	if not valid_file_list:
		raise RuntimeError("No valid .dtseries.nii files found matching metadata.")

	# 4) Log class distribution
	from collections import Counter
	ctr = Counter(label_map_by_relpath[rel][0] for rel in valid_file_list)
	logger.info("Class distribution: %s", dict(ctr))

	return valid_file_list, {rel: label_map_by_relpath[rel] for rel in valid_file_list}


# def load_adni_file_list(
#     data_dir: str,
#     metadata_path: str,
#     label_column: str,
# ) -> Tuple[List[str], Dict[str, Tuple[str, float, str]]]:
#     """
#     Recursively scan `data_dir` for all files ending in "parcellated.dtseries.nii". For each file:
#       • Extract the parent folder name (e.g. "I238623").
#       • Look up that `image_id` in the metadata CSV.

#     Returns:
#       • file_list: List of relative paths (under `data_dir`) to each valid file.
#       • label_map_by_relpath: Dict mapping relative_path → (researchGroup, subjectAge, subjectSex).
#     """

#     # Step 1: Read & sanitize the metadata CSV
#     logger.info(f"Loading metadata from '%s'", metadata_path)
#     meta = pd.read_csv(metadata_path)
#     meta.columns = [c.strip() for c in meta.columns]
#     logger.info("Metadata columns: %s", list(meta.columns))
#     total_rows = len(meta)

#     required_cols = {'filename', 'researchGroup', 'subjectAge', 'subjectSex'}
#     missing = required_cols - set(meta.columns)
#     if missing:
#         raise ValueError(f"Metadata CSV missing required columns: {missing}")

#     logger.info("Metadata rows before filtering: %d", total_rows)

#     # Build a map: image_id -> (researchGroup, subjectAge, subjectSex)
#     image_id_map: Dict[str, Tuple[str, float, str]] = {}
#     for _, row in tqdm(meta.iterrows(), total=total_rows, desc="Parsing metadata CSV", unit="row"):
#         fn = str(row['filename']).strip()
#         grp = str(row['researchGroup']).strip()
#         age = float(row['subjectAge'])
#         sex = str(row['subjectSex']).strip()

#         if label_column == 'researchGroup' and grp in {'SMC', 'Patient', 'EMCI', 'LMCI', 'AD'}:
#             continue

#         fn_base = os.path.splitext(fn)[0]
#         if '_' not in fn_base:
#             logger.warning("Skipping filename without underscore: %s", fn)
#             continue
#         image_id = fn_base.rsplit('_', 1)[-1]
#         image_id_map[image_id] = (grp, age, sex)

#     logger.info("Built image_id_map with %d entries", len(image_id_map))

#     # Step 2: Walk data_dir for files
#     logger.info("Scanning data_dir '%s' for .dtseries.nii files", data_dir)
#     all_dtseries: List[str] = []
#     for root, _, files in tqdm(os.walk(data_dir), desc="Walking data_dir", unit="dir"):
#         for fn in files:
#             if fn.endswith('parcellated.dtseries.nii'):
#                 full_path = os.path.join(root, fn)
#                 rel_path = os.path.relpath(full_path, data_dir)
#                 all_dtseries.append(rel_path)
#     logger.info("Found %d dtseries files under data_dir", len(all_dtseries))

#     # Step 3: Match files against metadata
#     valid_file_list: List[str] = []
#     label_map_by_relpath: Dict[str, Tuple[str, float, str]] = {}
#     skipped = 0
#     for rel_path in tqdm(all_dtseries, desc="Matching files to metadata", unit="file"):
#         parts = rel_path.split(os.sep)
#         if len(parts) < 2:
#             logger.warning("Skipping malformed path with too few parts: %s", rel_path)
#             skipped += 1
#             continue
#         image_id = parts[-2]
#         if image_id in image_id_map:
#             valid_file_list.append(rel_path)
#             label_map_by_relpath[rel_path] = image_id_map[image_id]
#         else:
#             skipped += 1
#     logger.info("Matched %d valid files, skipped %d files", len(valid_file_list), skipped)

#     if not valid_file_list:
#         raise RuntimeError(f"No valid '.dtseries.nii' files found under '{data_dir}' that match metadata image IDs.")

#     # Summary by group
#     from collections import Counter
#     grp_counts = Counter([v[0] for v in label_map_by_relpath.values()])
#     logger.info("Class distribution: %s", dict(grp_counts))

#     return valid_file_list, label_map_by_relpath

# def load_adni_file_list(
# 	data_dir: str,
# 	metadata_path: str,
# 	label_column: str,
# ) -> Tuple[List[str], Dict[str, Tuple[str, float]]]:
# 	"""
# 	Recursively scan `data_dir` for all files ending in “.dtseries.nii”. For each file:
# 	  • Extract the parent folder name (e.g. “I238623”).
# 	  • Look up that `image_id` in the metadata CSV.

# 	The metadata CSV (at `metadata_path`) must have at least these columns:
# 		- 'filename'       e.g. "ADNI_002_S_0295_Resting_State_fMRI_S110474_I238623.xml"
# 		- 'researchGroup'  e.g. "CN" or "AD"
# 		- 'subjectAge'     e.g. 90.0301

# 	We parse each CSV `filename` to obtain its trailing image ID (the part after the last “_”, 
# 	like “I238623”). We build a map:
# 		image_id → (researchGroup, subjectAge)

# 	Then, whenever we encounter a file whose parent directory is “I238623” (for example), 
# 	we know exactly which CSV row to grab from.

# 	Returns:
# 	  • file_list: List of relative paths (under `data_dir`) to each valid “*.dtseries.nii”.
# 	  • label_map_by_relpath: Dict mapping relative_path → (researchGroup, subjectAge).

# 	Raises:
# 	  • ValueError   if CSV is missing any of the required columns.
# 	  • RuntimeError if no valid “.dtseries.nii” files are found under data_dir.
# 	"""

# 	# ─── Step 1: Read & sanitize the metadata CSV ─────────────────────────────────
# 	meta = pd.read_csv(metadata_path)
# 	meta.columns = [c.strip() for c in meta.columns]

# 	required_cols = {'filename', 'researchGroup', 'subjectAge', 'subjectSex'}
# 	if not required_cols.issubset(set(meta.columns)):
# 		missing = required_cols - set(meta.columns)
# 		raise ValueError(f"Metadata CSV missing required columns: {missing}")

# 	# Build a map: image_id → (researchGroup, subjectAge)
# 	# Each CSV “filename” is something like:
# 	#   "ADNI_002_S_0295_Resting_State_fMRI_S110474_I238623.xml"
# 	# We want to extract the final "_I238623" portion (without ".xml") as the key.
# 	image_id_map: Dict[str, Tuple[str, float]] = {}

# 	for _, row in meta.iterrows():
# 		fn = str(row['filename']).strip()
# 		grp = str(row['researchGroup']).strip()
# 		age = float(row['subjectAge'])
# 		sex = str(row['subjectSex']).strip()
# 		if label_column == 'researchGroup':
# 			if grp == 'SMC' or grp == 'Patient' or grp == 'EMCI' or grp == 'LMCI' or grp == 'AD':
# 				continue
		
# 		# Strip off extension; then take everything after the final underscore to get "Ixxxxx"
# 		#    e.g. fn_base = "ADNI_002_S_0295_Resting_State_fMRI_S110474_I238623"
# 		fn_base = os.path.splitext(fn)[0]
# 		# The image ID is whatever follows the last '_', e.g. "I238623"
# 		if '_' not in fn_base:
# 			continue
# 		image_id = fn_base.split('_')[-1]
# 		image_id_map[image_id] = (grp, age, sex)
# 	# ─── Step 2: Walk data_dir recursively to find all “.dtseries.nii” ───────────────
# 	all_dtseries: List[str] = []
# 	for root, _, files in os.walk(data_dir):
# 		for fn in files:
# 			if fn.endswith('parcellated.dtseries.nii'):
# 				# Build the path relative to data_dir
# 				full_path = os.path.join(root, fn)
# 				rel_path = os.path.relpath(full_path, data_dir)
# 				all_dtseries.append(rel_path)
# 	# all_dtseries = list_parcellated_dtseries(data_dir)
# 	# ─── Step 3: For each dtseries, check if its parent folder is a known image_id ───
# 	valid_file_list: List[str] = []
# 	label_map_by_relpath: Dict[str, Tuple[str, float]] = {}

# 	for rel_path in all_dtseries:
# 		# Extract the immediate parent folder name of the dtseries file.
# 		# e.g. rel_path = 
# 		#   "002_S_0295/Resting_State_fMRI/2011-06-02_07_56_36.0/I238623/\
# 		#     Resting_State_fMRI_20110602075636_501_parcellated.dtseries.nii"
# 		#
# 		# We split on os.sep and take the second-to-last element:
# 		parts = rel_path.split(os.sep)
# 		# parts[-1] is the filename itself
# 		# parts[-2] should be the folder “I238623”
# 		if len(parts) < 2:
# 			# In theory this shouldn’t happen if the dtseries files are nested under at least one folder
# 			continue

# 		parent_folder = parts[-2]  # e.g. "I238623"
# 		image_id = parent_folder

# 		if image_id in image_id_map:
# 			valid_file_list.append(rel_path)
# 			label_map_by_relpath[rel_path] = image_id_map[image_id]
# 		else:
# 			# You can uncomment this to log skipped files:
# 			# print(f"[!] Skipping '{rel_path}' (parent folder '{image_id}' not in CSV).")
# 			pass

# 	if not valid_file_list:
# 		raise RuntimeError(
# 			f"No valid '.dtseries.nii' files found under '{data_dir}' that match metadata image IDs."
# 		)

# 	return valid_file_list, label_map_by_relpath


def load_adni_amyloid_file_list(
	dtseries_root: str,
	amyloid_csv: str
) -> Tuple[List[str], Dict[str, Tuple[int, float]]]:
	"""
	Reads `amyloid_csv` (must have columns PTID (relative_path), AV45, LABEL),
	and simply returns:
	  - file_list: List of relative paths under dtseries_root
	  - label_map: Dict mapping each relative_path → (label_int, av45_value)
	"""
	df = pd.read_csv(amyloid_csv)
	df.columns = df.columns.str.strip()
	required = {'PTID', 'AV45', 'LABEL'}
	if not required.issubset(df.columns):
		missing = required - set(df.columns)
		raise ValueError(f"CSV missing required columns: {missing}")

	root = Path(dtseries_root)

	all_files: List[str] = []
	label_map: Dict[str, Tuple[int, float]] = {}

	print(f"Validating {len(df)} entries against '{dtseries_root}'…")
	for _, row in tqdm(df.iterrows(), total=len(df), desc="Entries"):
		rel_path = str(row['PTID']).strip()
		full_path = root / rel_path
		new_full_path = str(full_path).replace(
			'_parcellated.dtseries.nii',
			'_450_parcellated.dtseries.nii'
		)
		full_path = Path(new_full_path)
		if not full_path.exists():
			tqdm.write(f"[WARN] File not found on disk: {rel_path}")
			continue

		lbl_str = str(row['LABEL']).strip().lower()
		label_int = 1 if lbl_str == 'positive' else 0
		av45 = float(row['AV45'])

		all_files.append(full_path)
		label_map[full_path] = (label_int, av45)

	if not all_files:
		raise RuntimeError(
			f"No files found under '{dtseries_root}' matching any PTID/paths in '{amyloid_csv}'"
		)

	print(f"\nSummary: {len(all_files)} files from {len(df)} CSV entries.")
	return all_files, label_map

# import pdb
# import subprocess

# def list_parcellated_dtseries(data_dir: str) -> List[str]:
# 	"""
# 	Uses the system `find` (much faster than os.walk) to collect
# 	all files ending in 'parcellated.dtseries.nii' under data_dir,
# 	returning their paths relative to data_dir.
# 	"""
# 	# Build and run the find command
# 	cmd = [
# 		"find", data_dir,
# 		"-type", "f",
# 		"-name", "*.parcellated.dtseries.nii"
# 	]
# 	try:
# 		output = subprocess.check_output(cmd, text=True)
# 	except subprocess.CalledProcessError as e:
# 		raise RuntimeError(f"Error running find: {e}")

# 	# Split into lines and relativize
# 	full_paths = output.strip().splitlines()
# 	return [os.path.relpath(p, data_dir) for p in full_paths]