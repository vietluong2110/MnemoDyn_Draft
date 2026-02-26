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
from glob import glob
import logging

logger = logging.getLogger(__name__)

import pdb

class Demo_Dataset(Dataset):
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

        if self.label_column == 'Age':
            label = self.labels[idx][0]
        else:
            label = self.labels[idx][1]
            if label == 1 and np.random.rand() < 0.5:
                x = x + 0.1 * torch.rand_like(x) 
        return x, coeffs, label, self.time_step

    def __len__(self) -> int:
        return len(self.data)

class DemoDataModule:
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
        if label_column == 'age' or 'gender':
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

        self.normalizer = Normalizer_update.from_statistics(stats, method=self.normalize)

        self.train_dataset = Demo_Dataset(self.data_dir, self.train_files, self.label_map, self.time_step, 
                                    self.interpol_method, self.one_channel, self.subset, self.dim_D, 
                                    self.target_length, self.normalizer, self.num_parcels, self.label_column)
        self.test_dataset = Demo_Dataset(self.data_dir, self.test_files, self.label_map, self.time_step, 
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

def load_demo_file_list(
    data_dir: str,
    metadata_path: str,
    label_column: str,   # 'age' or 'gender'/'sex'
    parcellate : int
) -> Tuple[List[str], Dict[str, Tuple[float, int]]]:
    """
    Load ds005747 parcellated dtseries paths and labels from participants.tsv.

    Returns:
      valid_file_list: list[str] of relative paths that exist under data_dir
      label_map: dict[rel_path] -> (age: float, sex: int) where sex M=1, F=0
    """
    if label_column.lower() not in {"age", "gender", "sex"}:
        raise ValueError("label_column must be one of: 'age', 'gender', 'sex'")

    logger.info("Loading participants metadata from %s", metadata_path)
    sep = "\t" if str(metadata_path).endswith(".tsv") else ","
    meta = pd.read_csv(metadata_path, sep=sep)
    meta.columns = [c.strip() for c in meta.columns]

    required = {"participant_id", "age", "gender"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"participants file is missing columns: {missing}")

    # Build participant -> (age, sex) map
    participant_map: Dict[str, Tuple[float, int]] = {}
    for _, row in meta.iterrows():
        pid = str(row["participant_id"]).strip()
        if not pid:
            continue

        try:
            age = float(row["age"])
        except Exception:
            continue

        sex_raw = str(row["gender"]).strip().upper()
        if sex_raw in {"M", "MALE", "1"}:
            sex = 1
        elif sex_raw in {"F", "FEMALE", "0"}:
            sex = 0
        else:
            continue

        participant_map[pid] = (age, sex)

    logger.info("Metadata entries after filtering: %d", len(participant_map))

    valid_file_list: List[str] = []
    label_map: Dict[str, Tuple[float, int]] = {}
    missing_files: List[str] = []

    for pid, labels in tqdm(participant_map.items(), desc="Matching files"):
        subj_dir = os.path.join(data_dir, pid)
        if not os.path.isdir(subj_dir):
            missing_files.append(f"{pid}: subject directory not found")
            continue

        candidates = sorted(glob(os.path.join(subj_dir, "*_parcellated.dtseries.nii")))
        if not candidates:
            missing_files.append(f"{pid}: no parcellated dtseries file")
            continue

        chosen = None
        for c in candidates:
            try:
                if int(nib.load(c).shape[1]) == int(parcellate):
                    chosen = c
                    break
            except Exception:
                continue

        if chosen is None:
            missing_files.append(
                f"{pid}: no parcellated file with {parcellate} parcels (found {len(candidates)} candidates)"
            )
            continue

        rel_path = os.path.relpath(chosen, data_dir)
        valid_file_list.append(rel_path)
        label_map[rel_path] = labels

    if missing_files:
        logger.warning(
            "Unmatched participants/files (%d). Showing up to 5:\n%s",
            len(missing_files),
            "\n".join(missing_files[:5]) + ("...\n" if len(missing_files) > 5 else "")
        )

    if not valid_file_list:
        raise RuntimeError("No valid parcellated .dtseries.nii files found from participants metadata.")

    logger.info("Valid files: %d", len(valid_file_list))
    return valid_file_list, label_map
