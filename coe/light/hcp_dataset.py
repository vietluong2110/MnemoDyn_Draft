# Prepare outputs
'''
Fast and clean implementation of HCP_Dataset with preloading, vectorized preprocessing,
and precomputed normalization and interpolation coefficients.
'''
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchcde
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import nibabel as nib
import numpy as np
from scipy.interpolate import interp1d


def load_cifti_dtseries(data_dir, rel_path):
	full_path_ap_1 = os.path.join(data_dir, rel_path)

	ap_dir, ap_file = os.path.split(rel_path)
	pa_dir  = ap_dir.replace('REST1_AP', 'REST1_PA')
	pa_file = ap_file.replace('REST1_AP', 'REST1_PA')
	pa_path = os.path.join(pa_dir, pa_file)

	full_path_pa_1 = os.path.join(data_dir, pa_path)

	# ap_dir, ap_file = os.path.split(rel_path)
	# pa_dir_2  = ap_dir.replace('REST1_AP', 'REST2_PA')
	# pa_file_2 = ap_file.replace('REST1_AP', 'REST2_PA')
	# pa_path = os.path.join(pa_dir_2, pa_file_2)

	# full_path_pa_2 = os.path.join(data_dir, pa_path)

	
	# ap_dir, ap_file = os.path.split(rel_path)
	# ap_dir_2  = ap_dir.replace('REST1_AP', 'REST2_AP')
	# ap_file_2 = ap_file.replace('REST1_AP', 'REST2_AP')
	# pa_path = os.path.join(ap_dir_2, ap_file_2)

	# full_path_ap_2 = os.path.join(data_dir, pa_path)

	# if not os.path.exists(full_path_pa_1):
	# 	return None 

	# if not os.path.isfile(full_path_ap_1):
	# 	return None 

	# if not os.path.exists(full_path_pa_2):
	# 	return None 

	# if not os.path.isfile(full_path_ap_2):
	# 	return None

	# ─── Load the .dtseries.nii via nibabel ───────────────────────────
	img_ap_1 = nib.load(full_path_ap_1)
	img_pa_1 = nib.load(full_path_pa_1)

	# img_ap_2 = nib.load(full_path_ap_2)
	# img_pa_2 = nib.load(full_path_pa_2)
	
	# sampled_indices = np.sort(np.random.choice(img_ap_1.shape[0], size=300, replace=False))
	arr_ap_1: np.ndarray = img_ap_1.get_fdata().astype(np.float32)[:200, ...]  # shape [T, D]
	# # sampled_indices = np.sort(np.random.choice(img_pa_1.shape[0], size=300, replace=False))
	arr_pa_1: np.ndarray = img_pa_1.get_fdata().astype(np.float32)[:290, ...]

	# # sampled_indices = np.sort(np.random.choice(img_ap_2.shape[0], size=300, replace=False))
	# arr_ap_2: np.ndarray = img_ap_2.get_fdata().astype(np.float32)[:300, ...]  # shape [T, D]
	# # sampled_indices = np.sort(np.random.choice(img_pa_2.shape[0], size=300, replace=False))
	# arr_pa_2: np.ndarray = img_pa_2.get_fdata().astype(np.float32)[:300, ...]


	arr = np.concatenate((arr_ap_1, arr_pa_1))
	# arr = np.concatenate((arr_ap_1))
	# arr = arr_ap_1
	return arr 

class HCPA_Dataset(Dataset):
	def __init__(
		self,
		data_dir: str,
		file_list,
		label_map, 

		time_step: float,

		interpol: str,
		one_channel: int = None,
		subset: bool = False,
		dim_D: list = None,
		target_length: int = 1200,
		normalizer = None, 
		num_parcels: int = 333,
	):
		# Store basic parameters
		self.tsv_dir = data_dir
		self.file_list = file_list
		self.label_map = label_map 
		self.time_step = time_step
		self.interpol_method = interpol
		self.target_length = target_length
		self.num_parcels = num_parcels
		self.one_channel = one_channel
		self.subset = subset
		self.dim_D = dim_D
		self.normalizer = normalizer

		# 1) Preload data arrays in a list
		raw_data = []
		labels   = []

		print(f"⏳ Preloading {len(self.file_list)} dtseries files from '{data_dir}'...")
		for rel_path in tqdm(self.file_list, desc="Preloading dtseries", unit="file"):
			
			arr = load_cifti_dtseries(data_dir, rel_path)
			if arr is None:
				continue  
			# arr = arr_ap
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
			if one_channel is not None:
				x_tensor = x_tensor[:, [one_channel]]
			elif subset and (dim_D is not None):
				x_tensor = x_tensor[:, :dim_D]
			# print(x_tensor.shape)
			# ─── Pad or trim to target_length ────────────────────────────────
			reps = int(np.ceil(target_length / x_tensor.shape[0]))
			x_tensor = x_tensor.repeat((reps, 1))[:target_length]
			raw_data.append(x_tensor)

			# ─── Lookup labels by relative path ─────────────────────────────
			# label_map was built so that
			#   label_map[rel_path] = (researchGroup, subjectAge)
			if rel_path not in self.label_map:
				raise KeyError(f"No label found for '{rel_path}'.")
			label = self.label_map[rel_path]
			labels.append(label)

		# Stack for normalization
		self.data = torch.stack(raw_data)  # shape [N, T, D_sel]
		# self.normalizer = Normalizer_update(normalize)
		# self.normalizer.fit(all_data)
		# # Transform and store data
		# self.data = self.normalizer.transform(all_data)

		# Precompute interpolation coefficients
		interp_fn = {
			'linear': torchcde.linear_interpolation_coeffs,
			'spline': torchcde.hermite_cubic_coefficients_with_backward_differences,
		}.get(interpol)
		if interp_fn is None:
			raise ValueError(f"Invalid interpolation: {interpol}")
		self.coeffs = interp_fn(self.data)
		# self.coeffs = [interp_fn(self.data[i]) for i in range(len(self.data))]
		self.labels = labels

	def __getitem__(self, idx: int):
		x = self.data[idx]            # [T, D_sel]
		coeffs = self.coeffs[idx]     # interpolation object
		y = self.labels[idx]
		return x, coeffs, y, self.time_step

	def __len__(self) -> int:
		return len(self.data)

class HCPADataModule:
	def __init__(self,
			   tsv_dir, 
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
			   ):
		self.time_step = torch.from_numpy(np.arange(0, duration, duration/original_length)).float()
		self.label_map = label_map
		self.tsv_dir = tsv_dir
		self.interpol_method = interpol
		self.target_length = target_length
		self.num_parcels = num_parcels
		self.one_channel = one_channel
		self.normalize = normalize
		self.subset = subset
		self.dim_D = dim_D
		self.train_files, self.test_files = train_test_split(file_list, test_size=0.2, random_state=42)
		print("Train/Test split: %d/%d", len(self.train_files), len(self.test_files))

	def compute_normalization_stats(self):
		n_total = 0
		sum_, sum_sq = None, None
		gmin, gmax = float('inf'), float('-inf')
		pvmin, pvmax = None, None
		all_samples = []

		for fn in tqdm(self.train_files, desc='Normalization stats'):
			arr = load_cifti_dtseries(self.tsv_dir, fn)
			if arr is None:
				continue  
			data = arr 
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

		# self.normalizer = Normalizer_update.from_statistics(stats, method=self.normalize)

		self.train_dataset = HCPA_Dataset(self.tsv_dir, self.train_files, self.label_map, self.time_step, 
									self.interpol_method, self.one_channel, self.subset, self.dim_D, 
									self.target_length, self.num_parcels)
		self.test_dataset = HCPA_Dataset(self.tsv_dir, self.test_files, self.label_map, self.time_step, 
									self.interpol_method, self.one_channel, self.subset, self.dim_D, 
									self.target_length, self.num_parcels)
		

	def train_dataloader(self, train_bs):
		return DataLoader(self.train_dataset, batch_size=train_bs, shuffle=True)

	def test_dataloader(self, test_bs):
		return DataLoader(self.test_dataset, batch_size=test_bs, shuffle = False)


def load_hcp_aging_parcellated_list(
	parcellation_csv: str,
	dtseries_root: str,
	label_column: str = "age_scaled",
):
	import os
	import pandas as pd

	df = pd.read_csv(parcellation_csv)
	df = df.dropna(subset=["relative_path", label_column])

	if label_column.lower() == "sex":
		df = df[df["sex"].isin(["M", "F"])].copy()
		df["sex"] = df["sex"].map({"M": 1, "F": 0})
	def all_four_exist(rel_path):
		base_dir, base_file = os.path.split(rel_path)
		variants = {
			"REST1_AP": rel_path,
			"REST1_PA": os.path.join(base_dir.replace("REST1_AP", "REST1_PA"), base_file.replace("REST1_AP", "REST1_PA")),
			"REST2_AP": os.path.join(base_dir.replace("REST1_AP", "REST2_AP"), base_file.replace("REST1_AP", "REST2_AP")),
			"REST2_PA": os.path.join(base_dir.replace("REST1_AP", "REST2_PA"), base_file.replace("REST1_AP", "REST2_PA")),
		}
		return all(os.path.exists(os.path.join(dtseries_root, path)) for path in variants.values())

	# Filter only rows where all 4 related files exist
	df = df[df["relative_path"].apply(all_four_exist)].reset_index(drop=True)

	# Normalize if needed
	if label_column == 'interview_age':
		df = df[(df[label_column] >= 40) & (df[label_column] <= 90)]
		df[label_column] = df[label_column].astype(int)

		age_mean =df[label_column].mean()
		age_std = df[label_column].std()
		print(f'age_mean: {age_mean}')
		print(f'age_std: {age_std}')
		df[label_column] = (df[label_column] - age_mean) / age_std
		df = df.copy()
		

	if label_column == 'neo2_score_ne':
		neo_max = df[label_column].mean()
		neo_min = df[label_column].std()
		print(f'neo_min: {neo_min}')
		print(f'neo_max: {neo_max}')
		df[label_column] = (df[label_column] - neo_max) / neo_min

	# Prepare outputs
	df["full_path"] = df["relative_path"].apply(
	lambda p: os.path.join(
		dtseries_root,
		p.replace(
			'_parcellated.dtseries.nii',
			'450_parcellated.dtseries.nii'
		)
	)
	)
	file_list = df["full_path"].tolist()
	label_map = dict(zip(df["full_path"], df[label_column]))
	return file_list, label_map
