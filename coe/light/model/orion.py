"""
Refactored Model file with memory optimizations and improved code structure
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from typing import List, Tuple, Optional
import torch.utils.checkpoint as checkpoint
import torch.utils.benchmark as benchmark

from pytorch_wavelets import DWT1DForward, DWT1DInverse
from .conv1d_optimize import VmapRegionConv1D
from .dense_layer import PerChannelDenseEinsum
import torchcde
import pdb
import time as sys_time

DEBUG = False  # Global toggle

def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

class WaveletProcessor(nn.Module):
    """Memory-efficient wavelet processing with gradient checkpointing"""
    
    def __init__(self, wave: str = 'db4', n_levels: int = 3):
        super().__init__()
        self.wave = wave
        self.n_levels = n_levels
        self.forward_wavelet = DWT1DForward(wave=wave, J=1, mode='periodization')
        self.inverse_wavelet = DWT1DInverse(wave=wave, mode='periodization')
    
    def decompose_with_checkpointing(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Wavelet decomposition with gradient checkpointing to save memory"""
        current_approx = x
        detail_coeffs = []
        approx_coeffs = []
        for level in range(self.n_levels):
            # Use gradient checkpointing for memory efficiency
            if self.training:
                current_approx, current_detail = checkpoint.checkpoint(
                    self._single_decompose, current_approx, use_reentrant=False
                )
            else:
                current_approx, current_detail = self._single_decompose(current_approx)
            dprint(current_detail.shape, current_approx.shape)
            detail_coeffs.append(current_detail)
            approx_coeffs.append(current_approx)
            
        return approx_coeffs, detail_coeffs
    
    def _single_decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single level wavelet decomposition"""
        approx, detail = self.forward_wavelet(x)
        return approx, detail[0]
    
    def reconstruct_level(self, approx: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        """Single level wavelet reconstruction"""
        return self.inverse_wavelet((approx, [detail]))

class ORionSeq(nn.Module):
    """
    Optimized sequence model
    """
    def __init__(self,
                 wave: str = 'db2',
                 D: int = 350,
                 D_out: int = 350,
                 dim_d: int = 50,
                 dim_k: int = 350,
                 h_rank: int = 64,
                 original_length: int = 1000,
                 num_classes: int = 10,
                 nonlinearity: str = 'relu',
                 n_levels: int = 3,
                 K_dense: int = 2,
                 K_LC: List[int] = None,
                 nb: int = 3,
                 dense_rank: int = 10,
                 LC_rank: int = 4,
                 num_sparse_LC: int = 10,
                 interpol: str = 'linear',
                 conv_bias: bool = True,
                 linear_bias: bool = True,
                 predict: bool = False,
                 masked_modelling: bool = False,
                 scale: bool = True,
                 use_mixed_precision: bool = True,
                 use_mRLoss = False):
        
        super().__init__()
        
        # Store configuration
        self.wave = wave
        self.dim_D = D
        self.dim_D_out = D_out
        self.dim_d = dim_d
        self.dim_k = dim_k
        self.h_rank = h_rank
        self.original_length = original_length
        self.n_levels = n_levels
        self.K_dense = K_dense
        self.K_LC = K_LC or [2] * n_levels
        self.nb = nb
        self.dense_rank = dense_rank
        self.LC_rank = LC_rank
        self.num_classes = num_classes
        self.num_sparse_LC = num_sparse_LC
        self.interpol = interpol
        self.conv_bias = conv_bias
        self.linear_bias = linear_bias
        self.scale = scale
        self.predict = predict
        self.masked_modelling = masked_modelling
        self.use_mixed_precision = use_mixed_precision
        self.use_mRLoss = use_mRLoss
        
        # Activation function
        self.activation = self._get_activation(nonlinearity)
        
        # Wavelet processor
        self.wavelet_processor = WaveletProcessor(wave, n_levels)
        
        # Main processing layers with bottleneck
        self.input_projection = nn.Linear(self.dim_D, self.dim_d, bias=linear_bias)
        self.bottleneck = nn.Linear(self.dim_d, self.h_rank, bias=linear_bias)
        self.feature_expansion = nn.Linear(self.h_rank, self.dim_D * self.dim_k, bias=linear_bias)
        
        # Get dimensions efficiently
        self.dense_dim, self.output_sizes = self._compute_dimensions()
        dprint(f"Dense dimension: {self.dense_dim}")
        print(f"Output sizes: {self.output_sizes}")
        
        # Dense processing layers
        self.dense_layers = nn.ModuleList([PerChannelDenseEinsum(dim_k=self.dim_k, dense_dim=self.dense_dim, bias=True)
                                            for _ in range(self.K_dense)])
        
        self.conv_layers = nn.ModuleList([nn.ModuleList([
                VmapRegionConv1D(in_channels=2*self.dim_k, out_channels=2*self.dim_k, kernel_size=self.nb, num_regions=self.num_sparse_LC, input_length=self.output_sizes[-level-1])
                        ]) for level in range(self.n_levels)])

        # Output projection
        self.output_projection = nn.Linear(self.dim_k, self.dim_D_out, bias=linear_bias)
        
        # Scaling
        if self.scale:
            self.channel_scales = nn.Parameter(torch.ones(self.dim_D_out))
        
        # Prediction head
        if self.predict:
            self.prediction_head = nn.Sequential(
                nn.Linear(self.dim_D_out, 64, bias=True),
                self.activation,
                nn.Dropout(0.1),
                nn.Linear(64, self.num_classes, bias=True)
            )
        
        self._initialize_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function"""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU()
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]
    
    def _compute_dimensions(self) -> Tuple[int, List[int]]:
        """Compute dimensions without storing intermediate tensors"""
        with torch.no_grad():
            x = torch.randn(6, 1, self.original_length)
            current_approx = x
            output_sizes = []
            for l in range(self.n_levels):
                current_approx, current_detail = self.wavelet_processor._single_decompose(current_approx)
                assert current_approx.shape[2] == current_detail.shape[2]
                output_sizes.append(current_approx.shape[2])
            output_sizes.reverse()
            return current_approx.size(2), output_sizes
            
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, seq: torch.Tensor, coeffs: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Forward pass with memory optimizations"""
        
        # Use mixed precision if enabled
        if self.use_mixed_precision and self.training:
            return self._forward_mixed_precision(seq, coeffs, time)
        else:
            return self._forward_full_precision(seq, coeffs, time)
    
    def _forward_full_precision(self, seq: torch.Tensor, coeffs: torch.Tensor, time: torch.Tensor):
        """Full precision forward pass"""
        time_step = time[0, :]
        batch_size, sequence_length = seq.shape[:2] 
        # Sequence 
        # Interpolation
        der_X = self._compute_derivative(coeffs, time_step)
        dprint("seq:", seq.shape)
        dprint("coeff:", coeffs.shape)
        dprint("time_step:", time_step.shape)
        dprint("der_X:", der_X.shape)

        # Feature extraction with bottleneck
        z = self.activation(self.input_projection(seq))
        dprint("z:",z.shape)
        z = self.activation(self.bottleneck(z))
        dprint("z:",z.shape)
        h = self.activation(self.feature_expansion(z))
        dprint("h:",h.shape)
        h = h.view(batch_size, sequence_length, self.dim_D, self.dim_k)
        dprint("h:",h.shape)
        
        # Compute velocity field
        v = torch.einsum('blkD,blDo->blko', h.transpose(2, 3), der_X).squeeze(-1)
        dprint("v:", v.shape)
        v = v.transpose(1, 2)
        dprint("v:", v.shape)
        
        # Wavelet decomposition
        approx_coeffs, detail_coeffs = self.wavelet_processor.decompose_with_checkpointing(v)
        
        # Process coarsest level
        current_approx = approx_coeffs[-1]
        dprint("Coarse:", current_approx.shape)
        for dense_layer in self.dense_layers:
            current_approx = self.activation(dense_layer(current_approx))
        dprint("Coarse:", current_approx.shape)

        dth_current_approx = current_approx
        ##This one 
        if self.use_mRLoss:
            mulit_res_approx_loss = 0

        # Reconstruct through levels
        for level in reversed(range(self.n_levels)):
            detail = detail_coeffs[level]
            approx = approx_coeffs[level]
            dprint("approx:", approx.shape)
            # Process level
            level_input = torch.cat([detail, approx], dim=1)
            dprint("level input:", level_input.shape)
            
            for conv_layer in self.conv_layers[level]:
                dprint("before:", level_input.shape)
                conv_output = conv_layer(level_input)
                level_input = self.activation(conv_output)
                dprint("after:", level_input.shape)
            # store ONCE per scale

            current_approx = dth_current_approx
            dprint("current approx:", current_approx.shape)
            current_approx = self.shape_correction(level_input, current_approx)
            dprint("Shape corrected current appox:", current_approx.shape)

            if self.use_mRLoss:
                dprint("ML loss:", approx_coeffs[level].shape)
                mulit_res_approx_loss += F.mse_loss(current_approx, approx_coeffs[level], reduction='mean')

            padded_current_approx = torch.cat([torch.zeros_like(current_approx), current_approx], dim = 1)
            dprint("padded current approx:", padded_current_approx.shape)
            X_l = padded_current_approx + level_input

            bs, dk, length = X_l.shape
            X_l_detail = X_l[:, :dk//2, :]
            X_l_approx = X_l[:, dk//2:, :]
            dprint("XL:", X_l.shape, X_l_approx.shape, X_l_detail.shape)
            ##Some of dth_current_approx here 
            dth_current_approx = self.wavelet_processor.reconstruct_level(X_l_approx, X_l_detail)
            dprint("CA:", dth_current_approx.shape)
        # Final processing
        dprint("pre final :", dth_current_approx.shape)
        output = self.output_projection(dth_current_approx.transpose(1,2))
        dprint("output:", output.shape)
        
        if self.scale:
            output = output * self.channel_scales.view(1, 1, -1)
 
        if self.predict:
            prediction = self.prediction_head(output[:, -1, :])
            return output, prediction
        
        # if self.use_mRLoss:
        #     return output, mulit_res_approx_loss
        return output
            
    @torch.cuda.amp.autocast()
    def _forward_mixed_precision(self, seq: torch.Tensor, coeffs: torch.Tensor, time: torch.Tensor):
        """Mixed precision forward pass"""
        return self._forward_full_precision(seq, coeffs, time)
    
    def _compute_derivative(self, coeffs: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        """Compute path derivative using interpolation"""
        with torch.no_grad():
            if self.interpol == 'linear':
                path_X = torchcde.LinearInterpolation(coeffs, time_step)
            elif self.interpol == 'spline':
                path_X = torchcde.CubicSpline(coeffs, time_step)
            else:
                raise ValueError(f"Invalid interpolation type: {self.interpol}")
        
            return path_X.derivative(time_step).unsqueeze(-1).float()
    
    def shape_correction(self, chi_l, current_approx):
        if chi_l.shape[-1] == current_approx.shape[-1]:
            return current_approx
        else:
            left_diff = chi_l.shape[-1] - current_approx.shape[-1]
            m = nn.ConstantPad1d((left_diff,0), 0)
            current_approx = m(current_approx)
            return current_approx
    
    def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate memory usage (rough approximation)
        param_memory = total_params * 4 / (1024**3)  # GB for float32
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params, 
            'estimated_param_memory_gb': param_memory
        }


def create_optimized_ORion_model(**kwargs) -> ORionSeq:
    """Create an optimized ORion model with sensible defaults"""
    defaults = {
        'wave': 'db2',
        'D': 333,
        'D_out': 333, 
        'dim_d': 150,
        'dim_k': 350,
        'h_rank': 5,
        'original_length': 1200,
        'num_classes': 1,
        'nonlinearity': 'tanh',
        'n_levels': 6,
        'K_dense': 32,
        'K_LC': [2, 2, 2, 2, 2, 2],
        'nb': 3,
        'dense_rank': 8,
        'LC_rank': 4,
        'num_sparse_LC': 5,
        'interpol': 'spline',
        'conv_bias': True,
        'linear_bias': True,
        'predict': False,
        'masked_modelling': False,
        'scale': False,
        'use_mixed_precision': False,
        'use_mRLoss': False
    }
    
    # Update defaults with user arguments
    config = {**defaults, **kwargs}
    return ORionSeq(**config)

def memory_hook(module, input, output):
    torch.cuda.synchronize()
    mem_bytes = torch.cuda.max_memory_allocated()
    mem_gb = mem_bytes / (1024 ** 3)
    module_name = module.__class__.__name__

    def extract_shapes(obj):
        if isinstance(obj, torch.Tensor):
            return obj.shape
        elif isinstance(obj, (list, tuple)):
            return [extract_shapes(o) for o in obj]
        elif isinstance(obj, dict):
            return {k: extract_shapes(v) for k, v in obj.items()}
        else:
            return str(type(obj))

    input_shapes = extract_shapes(input)
    output_shapes = extract_shapes(output)

    print(f"[{module_name}] Input(s): {input_shapes} | Output: {output_shapes} | Peak Mem: {mem_gb:.2f} GB")

# Global memory tracker
prev_mem = [0]

def delta_memory_hook(module, input, output):
    torch.cuda.synchronize()
    current_mem = torch.cuda.memory_allocated()
    delta = (current_mem - prev_mem[0]) / (1024 ** 3)  # GB
    prev_mem[0] = current_mem  # update for next layer

    module_name = module.__class__.__name__

    input_shapes = [inp.shape for inp in input if isinstance(inp, torch.Tensor)]
    if isinstance(output, (tuple, list)):
        output_shape = [out.shape for out in output if isinstance(out, torch.Tensor)]
    elif isinstance(output, torch.Tensor):
        output_shape = output.shape
    else:
        output_shape = str(type(output))

    if delta>1:
        print(f"[{module_name}] Input(s): {input_shapes} | Output: {output_shape} | ΔMem: {delta:.2f} GB")


class StackedORionModel(nn.Module):
    """
    Stack multiple ORionSeq layers with optional:
    - Automatic D_out -> D chaining
    - Residual connections
    """

    def __init__(self,
                 num_layers: int = 3,
                 defaults: Optional[dict] = None,
                 layer_kwargs: Optional[List[dict]] = None):
        super().__init__()

        if defaults is None:
            raise ValueError("You must provide a `defaults` dict.")

        self.use_residual = defaults.get('use_residual', False)
        self.layers = nn.ModuleList()
        self.use_mRLoss = defaults.get('use_mRLoss', False)

        prev_D_out = defaults["D_out"]
        for i in range(num_layers):
            # Layer-specific overrides
            overrides = layer_kwargs[i] if layer_kwargs and i < len(layer_kwargs) else {}

            # Automatically propagate D_out -> D for next layer
            if i > 0:
                overrides["D"] = prev_D_out
                overrides["interpol"] = "linear"  # <-- Set interpol to linear for layers > 0

            # Merge config and instantiate layer
            config = {**defaults, **overrides}
            config.pop('use_residual', None)  # <-- this line ensures it's removed
            layer = ORionSeq(**config)
            self.layers.append(layer)

            # Track D_out for chaining
            prev_D_out = config["D_out"]

    def forward(self, x, coeffs=None, time=None):
        if self.use_mRLoss:
            total_MRLoss = 0
        for i, layer in enumerate(self.layers):
            # Use provided coeffs for the first layer, x for the rest as these are linear interpolation and we don't need to compute coeff seperately which is expensive
            curr_coeffs = coeffs if i == 0 else x

            if self.use_mRLoss:
                out, mRloss = layer(x, coeffs=curr_coeffs, time=time)
                total_MRLoss += mRloss
            else:
                out = layer(x, coeffs=curr_coeffs, time=time)

            if self.use_residual and out.shape == x.shape:
                x = x + out
            else:
                x = out
        if self.use_mRLoss:
            return x, total_MRLoss
        else:
            return x

    
    def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate memory usage (rough approximation)
        param_memory = total_params * 4 / (1024**3)  # GB for float32
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params, 
            'estimated_param_memory_gb': param_memory
        }

def create_stacked_ORion_model(num_layers: int = 3,
                             layer_kwargs: Optional[List[dict]] = None,
                             **kwargs) -> StackedORionModel:
    """
    Factory function for StackedORionModel with:
    - shared defaults
    - optional per-layer overrides
    - automatic D_out -> D chaining
    - optional residual connections
    """
    defaults = {
        'wave': 'db2',
        'D': 333,
        'D_out': 333,
        'dim_d': 100,
        'dim_k': 200,
        'h_rank': 5,
        'original_length': 1200,
        'num_classes': 1,
        'nonlinearity': 'tanh',
        'n_levels': 6,
        'K_dense': 32,
        'K_LC': [2, 2, 2, 2, 2, 2],
        'nb': 3,
        'dense_rank': 8,
        'LC_rank': 4,
        'num_sparse_LC': 5,
        'interpol': 'spline',
        'conv_bias': True,
        'linear_bias': True,
        'predict': False,
        'masked_modelling': False,
        'scale': False,
        'use_mixed_precision': False,
        'use_residual': False,
        'use_mRLoss': True,
    }
    # Only keep keys from kwargs that are in defaults
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in defaults}
    filtered_kwargs['D_out'] = layer_kwargs[0]['D_out']
    base_config = {**defaults, **filtered_kwargs}
    return StackedORionModel(num_layers=num_layers,
                           defaults=base_config,
                           layer_kwargs=layer_kwargs)


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = create_optimized_ORion_model()
    # model =  model.to(device)

    layer_overrides = [{'D_out': 333},  {'D_out': 333},  {'D_out': 333},]
    model = create_stacked_ORion_model(
                num_layers=3,
                use_residual=True,
                layer_kwargs=layer_overrides)
    model = model.to(device)

    print("Model created successfully!")
    print(f"Memory usage: {model.get_memory_usage()}")
    
    # Reset peak memory tracker before forward
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Test forward pass
    batch_size, seq_len, dim_D = 4, 1200, 333
    seq = torch.randn(batch_size, seq_len, dim_D, device=device)
    coeffs = torch.randn(batch_size, 1199, 1332, device=device)
    # coeffs = torch.randn(batch_size, seq_len, dim_D, device=device)
    time = torch.linspace(0, 1, seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
    print(seq.shape, coeffs.shape, time.shape)
    skip_classes = (nn.Tanh, nn.ReLU, nn.Sigmoid, nn.Softmax, nn.SiLU, nn.LeakyReLU, nn.GELU, nn.Dropout)
    hooks = []
    torch.cuda.reset_peak_memory_stats()
    prev_mem[0] = torch.cuda.memory_allocated()

    # for name, module in model.named_modules():
    # 	# Skip containers (like Sequential), but keep layers
    # 	if len(list(module.children())) == 0 and not isinstance(module, skip_classes):
    # 		hook = module.register_forward_hook(memory_hook)
    # 		hooks.append(hook)
    # 		# hooks.append(module.register_forward_hook(delta_memory_hook))


    print("Starting forward pass")
    with torch.no_grad():
        start_time = sys_time.time()
        output, mRloss = model(seq, coeffs, time)
        print("No grad forwrad pass time:", sys_time.time() - start_time)
        print(f"Output shape: {output[0].shape if isinstance(output, tuple) else output.shape}")

    torch.cuda.synchronize()
    print(f"[Forward] Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"[Forward] Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    # print("\n"*10)

    for h in hooks:
        h.remove()
    hooks.clear()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    hooks = []
    torch.cuda.reset_peak_memory_stats()
    prev_mem[0] = torch.cuda.memory_allocated()

    # for name, module in model.named_modules():
    # 	# Skip containers (like Sequential), but keep layers
    # 	if len(list(module.children())) == 0 and not isinstance(module, skip_classes):
    # 		hook = module.register_forward_hook(memory_hook)
    # 		hooks.append(hook)
    # 		# hooks.append(module.register_forward_hook(delta_memory_hook))

    print("\n \n Starting backward pass")
    start_time = sys_time.time()
    output, mRloss = model(seq, coeffs, time)
    loss = F.mse_loss(output, seq, reduction='mean') + mRloss
    loss.backward()
    
    print("Loss", loss.item())
    print("Backward pass time:", sys_time.time() - start_time)
    print(f"Output shape: {output[0].shape if isinstance(output, tuple) else output.shape}")

    torch.cuda.synchronize()
    print(f"[Backward] Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"[Backward] Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")