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
from model.normalizer import Normalizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from scipy.interpolate import interp1d


logger = logging.getLogger(__name__)

import pdb

class UKBioBank_Dataset(Dataset):
    def __init__(
        self,
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
        file_list:       list of relative paths (under data_dir) to '*.dtseries.nii'
        label_map:       dict mapping each relative path → (researchGroup, subjectAge)
                         (obtained via load_adni_file_list).
        time_step:       passed through __getitem__
        interpol:        'linear' or 'spline'
        one_channel, subset, dim_D: same as before
        target_length:   target number of timepoints
        num_parcels:     expected number of parcels in that dtseries
        """
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

        # 1) Preload data arrays in a list
        raw_data = []
        labels   = []

        for rel_path in tqdm(self.file_list, desc="Preloading dtseries", unit="file"):
            full_path = rel_path
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
            raw_data.append(x_tensor)

            
            if rel_path not in self.label_map:
                raise KeyError(f"No label found for '{rel_path}'.")
            label = self.label_map[rel_path]
            labels.append(label)

        # Stack everything into a single tensor [N, T, D_sel]
        self.data = torch.stack(raw_data)

      

        # ─── Precompute interpolation coefficients ─────────────────────────
        interp_fn = {
            'linear': torchcde.linear_interpolation_coeffs,
            'spline': torchcde.hermite_cubic_coefficients_with_backward_differences,
        }.get(interpol)
        if interp_fn is None:
            raise ValueError(f"Invalid interpol='{interpol}'. Use 'linear' or 'spline'.")

        self.coeffs = interp_fn(self.data)

        self.labels = labels  # list of (group, age)

    def __getitem__(self, idx: int):
        x      = self.data[idx]       # [target_length, D_sel]
        coeffs = self.coeffs[idx]     # interpolation coefficients
        label = self.labels[idx]

        # noise_std = 0.01  # adjust scale of noise
        # noise = torch.randn_like(x) * noise_std
        # x_noisy = x + noise

        
        return x[:490, :], coeffs, label, self.time_step

    def __len__(self) -> int:
        return len(self.data)
    

class UKBioBankDataModule:
    def __init__(self,
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
        self.interpol_method = interpol
        self.target_length = target_length
        self.num_parcels = num_parcels
        self.one_channel = one_channel
        self.normalize = normalize
        self.subset = subset
        self.dim_D = dim_D
        self.label_column = label_column
        
        labels = [label_map[f][0] for f in file_list]
        self.train_files, self.test_files, train_labels, test_labels = train_test_split(file_list, labels, test_size=0.2, random_state=42)
    
        print("Train/Test split: %d/%d", len(self.train_files), len(self.test_files))

    def compute_normalization_stats(self):
        n_total = 0
        sum_, sum_sq = None, None
        gmin, gmax = float('inf'), float('-inf')
        pvmin, pvmax = None, None
        all_samples = []

        for fn in tqdm(self.train_files, desc='Normalization stats'):
            full_path = fn
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

        self.normalizer = Normalizer_update.from_statistics(stats, method=self.normalize)

        self.train_dataset = UKBioBank_Dataset(self.train_files, self.label_map, self.time_step, 
                                    self.interpol_method, self.one_channel, self.subset, self.dim_D, 
                                    self.target_length, self.normalizer, self.num_parcels, self.label_column)
        self.test_dataset = UKBioBank_Dataset(self.test_files, self.label_map, self.time_step, 
                                    self.interpol_method, self.one_channel, self.subset, self.dim_D, 
                                    self.target_length, self.normalizer, self.num_parcels, self.label_column)
        

    def train_dataloader(self, train_bs):
        return DataLoader(self.train_dataset, batch_size=train_bs, shuffle=True)

    def test_dataloader(self, test_bs):
        return DataLoader(self.test_dataset, batch_size=test_bs, shuffle = False)

def _to_int_or_none(x):
    try:
        xf = float(str(x).strip())
        if xf.is_integer():
            return int(xf)
    except Exception:
        pass
    return None

def load_ukbiobank_file_list(
    filename: str,
    metadata_path: str,
    label_column: str,   # 'gender' or 'dx'
) -> Tuple[List[str], Dict[str, Tuple[float, int, int]]]:
    """
    Load ADHD paths + labels with filtering by `label_column`.

    Expects CSV columns at least: path, gender, age, dx
    - If label_column == 'gender': keep rows with gender in {0,1}
    - If label_column == 'dx':     keep rows with dx in {0,1,2,3}

    Returns:
      valid_file_list: list[str] of relative paths that exist under data_dir
      label_map: dict[rel_path] -> (age: float, gender: int, dx: int)
    """


    logger.info("Loading metadata from %s", metadata_path)
    meta = pd.read_csv(metadata_path)[:1000]
    meta.columns = [c.strip() for c in meta.columns]



    # Build label map
    label_map_by_relpath: Dict[str, Tuple[float, int, int]] = {}
    for _, row in meta.iterrows():
        rel = str(row["path"]).strip()
        rel = os.path.join(rel, filename)
  
        d = int(row[label_column] == 'Male')
        label_map_by_relpath[rel] = d


    return list(label_map_by_relpath.keys()), label_map_by_relpath


def load_ukbiobank_file_list_age(
    filename: str,
    metadata_path: str,
    label_column: str,   # 'gender' or 'dx'
) -> Tuple[List[str], Dict[str, Tuple[float, int, int]]]:
    """
    Load ADHD paths + labels with filtering by `label_column`.

    Expects CSV columns at least: path, gender, age, dx
    - If label_column == 'gender': keep rows with gender in {0,1}
    - If label_column == 'dx':     keep rows with dx in {0,1,2,3}

    Returns:
      valid_file_list: list[str] of relative paths that exist under data_dir
      label_map: dict[rel_path] -> (age: float, gender: int, dx: int)
    """


    logger.info("Loading metadata from %s", metadata_path)
    meta = pd.read_csv(metadata_path)[:1000]
    meta.columns = [c.strip() for c in meta.columns]
    df = meta
    col = label_column
    stats = {
        "count": df[col].count(),
        "min": df[col].min(),
        "max": df[col].max(),
        "range": df[col].max() - df[col].min(),
        "median": df[col].median(),
        "mean": df[col].mean(),
        "std": df[col].std()
    }

    # meta[label_column] = (meta[label_column] - meta[label_column].median()) / (
    #     meta[label_column].quantile(0.75) - meta[label_column].quantile(0.25)
    # )
    # Build label map
    label_map_by_relpath: Dict[str, Tuple[float, int, int]] = {}
    for _, row in meta.iterrows():
        rel = str(row["path"]).strip()
        rel = os.path.join(rel, filename)
  
        d = row[label_column]
        label_map_by_relpath[rel] = [d]

    return list(label_map_by_relpath.keys()), label_map_by_relpath