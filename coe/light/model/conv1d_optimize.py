import math
import torch
import torch.nn.functional as F
from torch.func import vmap

def min_add_to_make_divisible(n, k):
    return (k - (n % k)) % k

# Module-level function — efficient and clean
def region_conv(x_r, w_r, b_r):
    return F.conv1d(x_r, w_r, b_r, stride=1, padding='same')

class VmapRegionConv1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_regions, input_length):
        super().__init__()
        
        assert input_length % num_regions == 0 or num_regions <= input_length
        self.num_regions = num_regions
        self.kernel_size = kernel_size
        self.C_in = in_channels
        self.C_out = out_channels
        self.input_length = input_length

        # Precompute padding
        self.padding_needed = (num_regions - (input_length % num_regions)) % num_regions
        self.T_padded = input_length + self.padding_needed
        self.T_per_region = self.T_padded // num_regions
        
        # Per-region weights
        self.weight = torch.nn.Parameter(torch.randn(num_regions, out_channels, in_channels, kernel_size))
        self.bias = torch.nn.Parameter(torch.randn(num_regions, out_channels))
    
    def forward(self, x):
        # x: (B, C_in, T)
        B, C_in, T = x.shape
        
        # Step 1: Pad only if necessary
        if self.padding_needed > 0:
            x = F.pad(x, (0, self.padding_needed))
        
        # Step 2: Sequential partition using view (more efficient) and  Step 3: Permute to (num_regions, B, C_in, T_per_region)
        x_regions = x.view(B, C_in, self.num_regions, self.T_per_region).permute(2, 0, 1, 3)
        
        # Step 4: Apply vmap over regions
        # out: (num_regions, B, C_out, T_per_region)
        out = vmap(region_conv, in_dims=(0, 0, 0))(x_regions, self.weight, self.bias)
        
        # Step 5: Concatenate regions back together (no loop)
        # from (R, B, C_out, T_per_region) → (B, C_out, T_padded)
        out = out.permute(1, 2, 0, 3).reshape(B, self.C_out, -1)
        
        # Step 6: Trim output to original time length
        if self.padding_needed > 0:
            out = out[:, :, :T]
        return out


import time as sys_time
if __name__ == "__main__":
    
    print("\n" + "="*50)
    model = VmapRegionConv1D(in_channels=2, out_channels=3, kernel_size=3, num_regions=5, input_length=504)
    x_test = torch.randn(4, 2, 504)
    print(f"Input shape: {x_test.shape}")
    out = model(x_test)
    print(f"Output shape: {out.shape}")

def check_view_vs_chunk():
    # First, let's verify both methods give the same result
    x = torch.arange(12).reshape(1, 1, 12)  # [0,1,2,3,4,5,6,7,8,9,10,11]
    num_regions = 3
    T_per_region = x.shape[2] // num_regions
    
    print("Original tensor:", x.squeeze())
    
    # Method 1: chunk
    chunks = torch.chunk(x, num_regions, dim=2)
    chunks_stacked = torch.stack(chunks, dim=0)
    print("Chunk method result shape:", chunks_stacked.shape)
    print("Chunk regions:")
    for i in range(num_regions):
        print(f"  Region {i}: {chunks_stacked[i].squeeze()}")
    
    # Method 2: view + transpose 
    x_view = x.view(1, 1, num_regions, T_per_region)
    x_view = x_view.transpose(0, 2).contiguous()  # (num_regions, 1, 1, T_per_region)
    print("\nView method result shape:", x_view.shape)
    print("View regions:")
    for i in range(num_regions):
        print(f"  Region {i}: {x_view[i].squeeze()}")
    
    # Check if they're equal
    print(f"\nAre results equal? {torch.equal(chunks_stacked, x_view)}")
    
    # Efficiency test
    print("\n" + "="*50)
    print("EFFICIENCY TEST")
    x_large = torch.randn(16, 64, 10000)  # Large tensor
    num_regions = 10
    num_trials = 1000
    
    start = sys_time.time()
    for _ in range(num_trials):
        chunks = torch.chunk(x_large, num_regions, dim=2)
        result_chunk = torch.stack(chunks, dim=0)
    chunk_time = sys_time.time() - start
    
    # Time view method  
    T_per_region = x_large.shape[2] // num_regions
    start = sys_time.time()
    for _ in range(num_trials):
        x_view = x_large.view(x_large.shape[0], x_large.shape[1], num_regions, T_per_region)
        result_view = x_view.transpose(0, 2).contiguous()
    view_time = sys_time.time() - start
    
    print(f"Chunk method: {chunk_time:.4f}s")
    print(f"View method: {view_time:.4f}s")
    print(f"View is {chunk_time/view_time:.2f}x faster" if view_time < chunk_time else f"Chunk is {view_time/chunk_time:.2f}x faster")