import os
import sys
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from tqdm import tqdm
import torchcde
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../../')
# from utils import analyze_model_parameters
from orion import create_stacked_ORion_model
from ema import EMACallback


import csv
import time as sys_time
import pdb
import torch.distributed as dist
import pprint
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from functools import partial
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.profilers import PyTorchProfiler
from torch.distributed.fsdp import MixedPrecision, CPUOffload, ShardingStrategy
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True


class Normalizer_update:
    def __init__(self, sample_data_list, method='standard'):
        self.scalers = []
        self.method = method
        self.global_stats = {}
        if len(sample_data_list) > 0:
            sample_data = torch.stack(sample_data_list)
            self.fit(sample_data)

    @classmethod
    def from_statistics(cls, stats, method='standard'):
        n = cls.__new__(cls)
        n.method = method
        n.global_stats = stats
        n.scalers = []  # not used in this case
        return n

    def transform_single(self, X):
        if X.dim() == 2:
            X = X.unsqueeze(0)
        X_np = X.numpy()
        X_transformed = np.zeros_like(X_np)
        method = self.method
        stats = self.global_stats

        if method == 'raw':
            X_transformed = X_np
        elif method == 'robust_scaler':
            X_transformed = X_np - X_np.mean(axis=1, keepdims=True)
            X_transformed = (X_transformed - stats['global_median']) / stats['global_iqr']
        elif method == 'all_patient_all_voxel':
            X_transformed = (X_np - stats['global_min']) / (stats['global_max'] - stats['global_min'])
        elif method == 'per_patient_all_voxel':
            for i in range(X_np.shape[0]):
                X_transformed[i] = (X_np[i] - X_np[i].min()) / (X_np[i].max() - X_np[i].min())
        elif method == 'per_patient_per_voxel':
            for i in range(X_np.shape[0]):
                for f in range(X_np.shape[2]):
                    v_min, v_max = X_np[i, :, f].min(), X_np[i, :, f].max()
                    if v_max > v_min:
                        X_transformed[i, :, f] = (X_np[i, :, f] - v_min) / (v_max - v_min)
        elif method == 'per_voxel_all_patient':
            for f in range(X_np.shape[2]):
                v_min, v_max = stats['per_voxel_min'][f], stats['per_voxel_max'][f]
                if v_max > v_min:
                    X_transformed[:, :, f] = (X_np[:, :, f] - v_min) / (v_max - v_min)
        elif method == 'subtract_mean':
            X_transformed = X_np - X_np.mean(axis=1, keepdims=True)
        elif method == 'subtract_mean_global_std':
            X_transformed = (X_np - X_np.mean(axis=1, keepdims=True)) / stats['global_std']
        elif method == 'subtract_mean_99th_percentile':
            X_transformed = (X_np - X_np.mean(axis=1, keepdims=True)) / stats['global_99th']
        return torch.tensor(X_transformed, dtype=torch.float32)

class HCPParcellatedDataset(Dataset):
    def __init__(self, subject_dirs, file_name, time_step, interpol, D, subset=False, one_channel=None, normalizer=None, run_token="default"):
        self.subject_dirs = subject_dirs
        self.file_name = file_name
        self.time_step = time_step
        self.interpol = interpol
        self.D = D
        self.subset = subset
        self.one_channel = one_channel
        self.normalizer = normalizer
        self.run_token = run_token
        
        # Validate subjects thoroughly during initialization
        self.valid_subjects = self._get_valid_subjects()
        print(f"Found {len(self.valid_subjects)} valid subjects out of {len(subject_dirs)} total")
        
        if len(self.valid_subjects) == 0:
            raise ValueError("No valid subjects found! Check your data paths and parameters.")

        self.coeffs_method = 'linear' if interpol == 'linear' else 'spline'

    def _get_valid_subjects(self):
        """Get subjects that both exist and can be loaded successfully."""
        valid_subjects = []
        failed_subjects = []
        
        for subject_dir in self.subject_dirs:
            file_path = os.path.join(subject_dir, self.file_name)
            
            # First check if file exists
            if not os.path.exists(file_path):
                failed_subjects.append((subject_dir, "File not found"))
                continue
            
            # Then try to actually load and validate the data
            try:
                data = self._load_and_validate_data(file_path)
                if data is not None:
                    valid_subjects.append(subject_dir)
                else:
                    failed_subjects.append((subject_dir, "Invalid data shape"))
            except Exception as e:
                failed_subjects.append((subject_dir, f"Loading error: {str(e)}"))
        
        # Log failed subjects for debugging
        if failed_subjects:
            print(f"Failed to load {len(failed_subjects)} subjects:")
            for subject, reason in failed_subjects:  
                print(f"  {os.path.basename(subject)}: {reason}")
        
        return valid_subjects

    def _load_and_validate_data(self, file_path):
        """Load data and validate its shape without applying normalization."""
        data = nib.load(file_path).get_fdata()
        
        if self.one_channel is not None and self.one_channel >= 0:
            if self.one_channel >= data.shape[1]:
                return None  # Channel index out of bounds
            data = np.expand_dims(data[:, self.one_channel], -1)
            expected_shape = (1200, 1)
        else:
            if self.subset:
                if self.D > data.shape[1]:
                    return None  # Not enough channels for subset
                data = data[:, :self.D]
            expected_shape = (1200, self.D)
        
        # Validate shape
        if data.shape != expected_shape:
            return None
            
        return data

    def _load_single_subject(self, idx):
        """Load and preprocess a single subject's data."""
        file_path = os.path.join(self.valid_subjects[idx], self.file_name)
        
        # We know this will work because we validated it in __init__
        data = self._load_and_validate_data(file_path)
        
        # Convert to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)
        
        # Apply normalization if provided
        if self.normalizer is not None:
            data_tensor = self.normalizer.transform_single(data_tensor.unsqueeze(0)).squeeze(0)

        return data_tensor

    def __len__(self):
        return len(self.valid_subjects)

    def __getitem__(self, idx):
        if idx >= len(self.valid_subjects):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.valid_subjects)}")
        
        # Load data (this should never fail since we pre-validated)
        data = self._load_single_subject(idx)
        
        # Compute interpolation coefficients
        data_batch = data.unsqueeze(0)
        with torch.no_grad():
            if self.coeffs_method == 'linear':
                coeffs = torchcde.linear_interpolation_coeffs(data_batch).squeeze(0)
            else:
                coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(data_batch).squeeze(0)

        return data, coeffs, self.time_step

class GordonHCPDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.args = argparse.Namespace(**kwargs)
        self.data_dir = "/mnt/sourav/GordonHCP/HCP_Dense"
        # self.filename = "rfMRI_REST1_LR_Atlas_hp2000_clean_Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3.dlabel_parcellated.dtseries.nii"
        self.filename = "rfMRI_REST1_LR_Atlas_hp2000_clean_Gordon333_FreesurferSubcortical.32k_fs_LR.dtseries_parcellated.dtseries.nii"

        self.time_step = torch.from_numpy(np.arange(0, self.args.duration, self.args.duration/self.args.original_length)).float()
        self.subject_dirs = [os.path.join(self.data_dir, d) for d in os.listdir(self.data_dir)
                             if os.path.isdir(os.path.join(self.data_dir, d))]
        self.train_dirs, self.val_dirs = train_test_split(self.subject_dirs, test_size=0.2, random_state=self.args.seed)

    def compute_normalization_stats(self):
        n_total = 0
        sum_, sum_sq = None, None
        gmin, gmax = float('inf'), float('-inf')
        pvmin, pvmax = None, None
        all_samples = []
        for sdir in tqdm(self.train_dirs, desc='Normalization stats'):
            fpath = os.path.join(sdir, self.filename)
            if not os.path.exists(fpath): continue
            try:
                data = nib.load(fpath).get_fdata()
                if self.args.one_channel >= 0:
                    data = np.expand_dims(data[:, self.args.one_channel], -1)
                    if data.shape != (1200, 1): continue
                else:
                    if self.args.subset:
                        data = data[:, :self.args.D]
                    if data.shape != (1200, self.args.D): continue
                
                t = torch.tensor(data, dtype=torch.float32)
                all_samples.append(t)

                n_total += t.shape[0]
                sum_ = sum_ + t.sum(0) if sum_ is not None else t.sum(0)
                sum_sq = sum_sq + (t ** 2).sum(0) if sum_sq is not None else (t ** 2).sum(0)
                pvmin = torch.min(pvmin, t.min(0).values) if pvmin is not None else t.min(0).values
                pvmax = torch.max(pvmax, t.max(0).values) if pvmax is not None else t.max(0).values
                gmin, gmax = min(gmin, t.min().item()), max(gmax, t.max().item())

            except Exception as e:
                print(f"Failed {fpath}: {e}")

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


    def setup(self, stage=None):
        run_token = f"seed{self.args.seed}_norm{self.args.normalize}_interp{self.args.interpol}"
        stats = self.compute_normalization_stats()
        # Optional: Pretty-print normalization stats
        if stats:
            print(f"[Rank 0] Normalization stats for method '{self.args.normalize}':")
            # pprint.pprint(stats)
        else:
            print(f"[Rank 0] Error: No normalization stats computed.")

        # Sanity check for required keys
        required_keys = {
            'robust_scaler': ['global_median', 'global_iqr'],
            'subtract_mean_global_std': ['global_std'],
            'subtract_mean_99th_percentile': ['global_99th'],
            'per_voxel_all_patient': ['per_voxel_min', 'per_voxel_max'],
            'all_patient_all_voxel': ['global_min', 'global_max'],
        }
        for key, required in required_keys.items():
            if self.args.normalize == key:
                for rk in required:
                    if rk not in stats:
                        raise ValueError(f"Missing key '{rk}' in computed stats for normalization method '{key}'")

        self.normalizer = Normalizer_update.from_statistics(stats, method=self.args.normalize)

        if stage in ('fit', None):
            self.train_dataset = HCPParcellatedDataset(self.train_dirs, self.filename, self.time_step,
                                                       self.args.interpol, self.args.D, self.args.subset,
                                                       self.args.one_channel, self.normalizer, run_token=run_token)
            self.val_dataset = HCPParcellatedDataset(self.val_dirs, self.filename, self.time_step,
                                                     self.args.interpol, self.args.D, self.args.subset,
                                                     self.args.one_channel, self.normalizer, run_token=run_token)
        if stage in ('test', None):
            self.test_dataset = HCPParcellatedDataset(self.val_dirs, self.filename, self.time_step,
                                                      self.args.interpol, self.args.D, self.args.subset,
                                                      self.args.one_channel, self.normalizer, run_token=run_token)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.train_bs, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.valid_bs, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.test_bs, num_workers=1)

class LitORionModelOptimized(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.args = argparse.Namespace(**kwargs)
        layer_overrides = [{'D_out': v} for v in self.args.D_out]
        self.model = create_stacked_ORion_model(num_layers=len(layer_overrides),
                        layer_kwargs=layer_overrides,
                        **vars(self.args))
        

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss(reduction='mean', delta=self.args.huber_delta)
        self.loss_type = self.args.loss_type

    def configure_optimizers(self):
        # Use AdamW with better weight decay handling
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.args.lr, 
            weight_decay=0.01,  # Slightly higher weight decay
            betas=(0.9, 0.95)   # Better betas for transformer-like models
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=self.args.lr * 0.01)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def forward(self, x, coeff, time):
        return self.model(x, coeff, time)
    
    def training_step(self, batch, batch_idx):
        # More frequent memory cleanup for large models
        if batch_idx % 25 == 0:  # More frequent than 50
            torch.cuda.empty_cache()
        x, coeff, time = batch
        # Use gradient checkpointing for memory efficiency during forward pass
        with torch.cuda.amp.autocast(enabled=self.trainer.precision == '16-mixed'):
            if self.args.use_mRLoss:
                x_pred, mRloss = self(x, coeff, time)
            else:
                x_pred = self(x, coeff, time)
                
        if self.args.use_mRLoss:
            mse = self.mse_loss(x_pred, x) + mRloss
        else:
            mse = self.mse_loss(x_pred, x)

        mae = self.mae_loss(x_pred, x)
        huber = self.huber_loss(x_pred, x)

        # CRITICAL: Delete x_pred immediately
        del x_pred
        torch.cuda.empty_cache()  # Force cleanup
        
        if self.loss_type == 'mse':
            loss = mse + mae
        elif self.loss_type == 'total':
            loss = mse + mae
        elif self.loss_type == 'mae':
            loss = mae
        elif self.loss_type == 'huber':
            loss = huber
        else:
            loss = mse + mae
            
        self.log("train_mse", mse.detach(), prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log("train_mae", mae.detach(), prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log("train_loss", loss.detach(), prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, coeff, time = batch
        with torch.cuda.amp.autocast(enabled=self.trainer.precision == '16-mixed'):
            if self.args.use_mRLoss:
                x_pred, mRloss = self(x, coeff, time)
            else:
                x_pred = self(x, coeff, time)
        if self.args.use_mRLoss:
            mse = self.mse_loss(x_pred, x) + mRloss
        else:
            mse = self.mse_loss(x_pred, x)
        mae = self.mae_loss(x_pred, x)
        huber = self.huber_loss(x_pred, x)

        # CRITICAL: Delete x_pred immediately
        del x_pred
        torch.cuda.empty_cache()  # Force cleanup
        
        if self.loss_type == 'mse':
            loss = mse + mae
        elif self.loss_type == 'total':
            loss = mse + mae
        elif self.loss_type == 'mae':
            loss = mae
        elif self.loss_type == 'huber':
            loss = huber
        else:
            loss = mse + mae
            
        self.log("val_mae", mae.detach(), prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_mse", mse.detach(), prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_loss", loss.detach(), prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, coeff, time = batch
        return self(x, coeff, time)

def configure_fsdp_strategy_improved(model_size_mb=None):
    """
    Improved FSDP strategy that actually shards parameters across GPUs
    """
    
    # More aggressive parameter wrapping for better sharding
    # Lower threshold means more modules get wrapped and sharded
    min_params_to_wrap = 1000  # Much lower than your current 100000
    
    # Use FULL_SHARD for maximum memory efficiency
    sharding_strategy = ShardingStrategy.FULL_SHARD
    
    # Enable mixed precision to save memory (if your model supports it)
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,      # Parameters in fp16
        reduce_dtype=torch.float16,     # Gradients in fp16  
        buffer_dtype=torch.float32,     # Buffers stay in fp32 for stability
    )
    
    # Enable CPU offload for parameters not currently being used
    cpu_offload = CPUOffload(offload_params=True)
    
    fsdp_strategy = pl.strategies.FSDPStrategy(
        # More aggressive wrapping policy
        auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=min_params_to_wrap),
        
        # Full sharding for maximum memory savings
        sharding_strategy=sharding_strategy,
        
        # Enable mixed precision
        mixed_precision=mixed_precision_policy,
        
        # CPU offload to save GPU memory
        cpu_offload=cpu_offload,
        
        # Enable activation checkpointing to trade compute for memory
        activation_checkpointing_policy={nn.Linear, nn.Conv1d, nn.Conv2d},
        
        # Full state dict for easier checkpointing
        state_dict_type="full",
        
        # Limit concurrent all-gathers to save memory
        limit_all_gathers=True,
        
        # Use backward prefetch for better performance
        backward_prefetch="BACKWARD_PRE",
        
        # Forward prefetch for better performance  
        forward_prefetch=True,
    )
    
    return fsdp_strategy

def get_adaptive_strategy(pytorch_total_params, num_gpus):
    """
    Choose strategy based on model size and number of GPUs
    """
    print(f"Model parameters: {pytorch_total_params:,}")
    print(f"Number of GPUs: {num_gpus}")
    
    # Calculate approximate model size in MB (assuming fp32)
    model_size_mb = (pytorch_total_params * 4) / (1024 * 1024)
    print(f"Estimated model size: {model_size_mb:.1f} MB")
    # return "ddp_find_unused_parameters_true"
    return "ddp"

    # if num_gpus == 1:
    # 	return None  # Use default single GPU
    # elif pytorch_total_params < 1_000_000:  # < 1M params
    # 	print("Using DDP for small model")
    # 	return "ddp"
    # else:
    # 	print("Using improved FSDP for larger model")
    # 	return configure_fsdp_strategy_improved(model_size_mb)

# Alternative: Custom wrapping policy for your specific model architecture
def create_custom_wrap_policy():
    """
    Create a custom wrapping policy based on your model architecture
    This gives you more control over what gets sharded
    """
    
    def custom_auto_wrap_policy(module, recurse, nonwrapped_numel):
        # Wrap any module with more than 1000 parameters
        if nonwrapped_numel >= 1000:
            return True
        # Always wrap these specific layer types
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)):
            return True
        return False
    
    return custom_auto_wrap_policy

# Memory optimization utilities
def optimize_memory_usage():
    """
    Additional memory optimization techniques
    """
    # Enable memory efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except:
        pass
    
    # Set memory fraction to leave some GPU memory free
    if torch.cuda.is_available():
        # Reserve 10% of GPU memory for other processes
        torch.cuda.set_per_process_memory_fraction(0.90)
    
    # Enable cudnn benchmarking for consistent input sizes
    torch.backends.cudnn.benchmark = True
    
    # Use memory efficient attention patterns
    torch.backends.cuda.enable_math_sdp(True)

# Gradient accumulation helper
def calculate_effective_batch_size(base_batch_size, num_gpus, grad_accum_steps):
    """
    Calculate effective batch size across GPUs and gradient accumulation
    """
    effective_batch_size = base_batch_size * num_gpus * grad_accum_steps
    print(f"Effective batch size: {base_batch_size} * {num_gpus} GPUs * {grad_accum_steps} accum = {effective_batch_size}")
    return effective_batch_size

# Updated training configuration
def get_optimized_trainer_config(args, num_gpus, pytorch_total_params):
    """
    Get optimized trainer configuration for memory efficiency
    """
    
    # Choose strategy based on model size
    strategy = get_adaptive_strategy(pytorch_total_params, num_gpus)
    
    # Adjust gradient accumulation based on available memory
    if num_gpus > 1 and pytorch_total_params > 1_000_000:
        # More aggressive gradient accumulation for large models
        grad_accum_steps = max(4, 8 // num_gpus)
    else:
        grad_accum_steps = 1
    
    # Calculate effective batch size
    effective_bs = calculate_effective_batch_size(args.train_bs, num_gpus, grad_accum_steps)
    print(f"Using gradient accumulation steps: {grad_accum_steps}")
    
    trainer_config = {
        'strategy': strategy,
        'accumulate_grad_batches': grad_accum_steps,
        # 'precision': "16-mixed" if num_gpus > 1 else "32-true",  # Use mixed precision for multi-GPU
        'precision': "32-true",  # Use mixed precision for multi-GPU
        'gradient_clip_val': 1.0,  # Gradient clipping for stability for large model
        'enable_model_summary': False,  # Disable to save memory
        # 'sync_batchnorm': num_gpus > 1,  # Sync batch norm across GPUs
    }
    return trainer_config
    

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for ORion")
    parser.add_argument('--original_length', type=int, default=1200, help='Total sequence length in time series')
    parser.add_argument('--dataset_type', type=str, default='GordonHCP')
    parser.add_argument('--storage', type=str, default='/nas/sourav/Result/Orion_333/')
    parser.add_argument("--seed", default=34, type=int, help="Setting seed for the entire experiment")
    parser.add_argument("--exp", default='hyperparameter_tuning', help="Adjusted in code: Experiment folder name")
    parser.add_argument('--wave', default='db2', type=str, help='pywavelet')
    parser.add_argument('--h_rank', default=5, type=int, help="Dimension of bottle neck layer, keep it small, make parameter efficient")
    parser.add_argument('--D', default=333, type=int, help="Dimension of observable variable")
    parser.add_argument("--D_out",
                        nargs="+",  # Accepts one or more arguments
                        type=int,   # Converts each argument to an integer
                        default=[333, 333, 333, 333],  # Default value if not provided
                        help="List of integers (e.g., --D_out 2 1 2 1)"
                        )
    parser.add_argument('--subset', action='store_true', default=False, help="Use first D channel only")
    parser.add_argument('--one_channel', default=-1, type=int, help="Use only this dimension")
    parser.add_argument('--dim_d', default=150, type=int, help="Latent dimension of evolution")
    parser.add_argument('--dim_k', default=350, type=int, help="Dimension of h_theta (first)")
    parser.add_argument('--num_classes', default=1, type=int, help="Output dimensionality of model")
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='Non linearity to use')
    parser.add_argument('--n_levels', default=6, type=int, help="Number of levels of wavelet decomposition")
    parser.add_argument('--K_dense', default=32, type=int, help="Number of dense layers")
    parser.add_argument('--dense_rank', default=8, type=int, help="Number of dense layers")
    parser.add_argument('--LC_rank', default=4, type=int, help="Number of dense layers")
    parser.add_argument('--nb', default=3, type=int, help="Diagonal banded length, determine the kernel size")
    parser.add_argument('--num_sparse_LC', default=5, type=int, help="Number of sparse LC unit")
    parser.add_argument('--conv_bias', action='store_false', default=True, help="Turn off bias in convolution layer")
    parser.add_argument('--interpol', default='spline', type=str, help='Interpolation type to use')
    parser.add_argument('--train_bs', default=8, type=int, help='Batchsize for train loader')
    parser.add_argument('--valid_bs', default=16, type=int, help='Batchsize for valid loader')
    parser.add_argument('--test_bs', default=16, type=int, help='Batchsize for test loader')
    parser.add_argument('--epoch', default=500, type=int, help='Number of epochs to train')
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate for the ORion model")
    parser.add_argument('--model_pred_save_freq', default=100, type=int, help='Saving frequency of model prediction')
    parser.add_argument('--normalize', default='robust_scaler', type=str, choices=['standard', 'minmax', 'raw', 'robust_scaler', 'all_patient_all_voxel', 'per_patient_all_voxel',
                        'per_patient_per_voxel', 'per_voxel_all_patient', 'subtract_mean', 'subtract_mean_global_std', 'subtract_mean_99th_percentile'],
                        help='Normalization type: standard (zero mean, unit variance) or minmax (0-1 range)')
    parser.add_argument('--loss_type', default='total', type=str, choices=['total', 'mae', 'mse', 'huber'],
                        help='Loss funcion MSE, MAE or both')
    parser.add_argument('--huber_delta', default=1.0, type=float, help="threhsold for huber loss function")
    parser.add_argument('--scale', action='store_true', help="Enable channel-wise scaling (default: off)")
    parser.add_argument('--predict', action='store_true', default=False, help="Turn on classification task")
    parser.add_argument('--masked_modelling', action='store_true', default=False, help="Turn on masked modelling")
    parser.add_argument('--profile', action='store_true', default=False, help="Use pytorch profiling")
    parser.add_argument("--nolinear_bias", action="store_false", dest="linear_bias", default=True, help="Disable the bias in linear layer")
    parser.add_argument("--K_LC",
                        nargs="+",  # Accepts one or more arguments
                        type=int,   # Converts each argument to an integer
                        default=[2, 2, 2, 2, 2, 2],  # Default value if not provided
                        help="List of integers (e.g., --K_LC 2 1 2 1)"
                        )
    parser.add_argument('--duration', default=1, type=float, help="Time duration of time series")
    parser.add_argument('--use_mixed_precision', action='store_true', default=False, help="Use mixed precision training")
    parser.add_argument('--use_residual', action='store_true', default=False, help="Use residual connection")
    parser.add_argument('--use_mRLoss', action='store_true', default=False, help="Use mulit-resolution losse")
    # parser.add_argument('--cpu_offload', action='store_true', default=False, help="Offload parameters to CPU")
    # parser.add_argument('--grad_accum_auto', action='store_true', default=True, help="Auto-adjust gradient accumulation")
        
    return parser.parse_args()


def main():
    torch.set_float32_matmul_precision("medium")
    print("ORion with Improved Multi-GPU Support!!")
    args = parse_args()
    num_GPU = torch.cuda.device_count()
    
    optimize_memory_usage()
    pl.seed_everything(args.seed, workers=True)
    print(args.D)
    dm = GordonHCPDataModule(**vars(args))
    try:
        model = LitORionModelOptimized(**vars(args))
    except Exception as e:
        print(f"❌ Skipping config due to error: {vars(args)}")
        print(f"Error: {e}")
        return
    # seq = torch.rand((16, 1200, 333))
    # coeffs = torch.rand((16, 1199, 1332))
    # time = torch.rand((1, 1200))
    # model = torch.compile(model, mode='reduce-overhead')
    # model = model.float()
    # model(seq, coeffs, time)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {pytorch_total_params:,}")
    
    trainer_config = get_optimized_trainer_config(args, num_GPU, pytorch_total_params)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_mae",
        mode="min",
        save_top_k=2,
        filename="model-{epoch:02d}-{val_mae:.4f}",
        verbose=True,
    )

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_mae",
        min_delta=0.0001,
        mode="min",
        patience=30,
        verbose=True,
        check_finite=True,
    )
    
    progress_bar = pl.callbacks.TQDMProgressBar(refresh_rate=10)
    
    ema_callback = EMACallback(decay=0.9)

    # Setup logging
    LOG_ROOT = args.storage
    EXP_NAME = args.exp + "_" + args.dataset_type
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=LOG_ROOT, name=EXP_NAME)
    csv_logger = pl.loggers.CSVLogger(save_dir=LOG_ROOT, name=EXP_NAME, version=tb_logger.version)

    # Configure trainer with optimized settings
    trainer = pl.Trainer(
        benchmark=False,
        deterministic=True,
        fast_dev_run=False,
        logger=[tb_logger, csv_logger],
        accelerator="gpu",
        devices=num_GPU,
        max_epochs=args.epoch,
        log_every_n_steps=10,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, early_stopping, progress_bar, ema_callback],
        check_val_every_n_epoch=1,
        **trainer_config  # Apply optimized configuration
    )
    
    # Add profiler only if requested
    if args.profile:
        profiler = PyTorchProfiler(
            dirpath=LOG_ROOT, 
            filename="debug",
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                dir_name=os.path.join(tb_logger.log_dir, "tb_profiler")
            ),
            schedule=torch.profiler.schedule(skip_first=1, wait=1, warmup=1, active=5),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        trainer.profiler = profiler

    # Print memory info before training
    if torch.cuda.is_available():
        print(f"Memory allocated (MB): {torch.cuda.memory_allocated() / 1e6:.1f}")
        print(f"Max memory allocated (MB): {torch.cuda.max_memory_allocated() / 1e6:.1f}")
        print(f"Memory reserved (MB): {torch.cuda.memory_reserved() / 1e6:.1f}")

    # Start training
    trainer.fit(model, dm)
    print("#####################################################")


if __name__ == '__main__':
    main()