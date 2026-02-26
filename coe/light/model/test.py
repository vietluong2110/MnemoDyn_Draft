import os, sys, pdb, argparse, yaml
import pandas as pd
import numpy as np
from tabulate import tabulate
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
# from main import GordonHCPDataModule, LitORionModelOptimized
from coe.light.model.main import GordonHCPDataModule, LitORionModelOptimized
from tqdm import tqdm 
from huggingface_hub import hf_hub_download

def process_and_print_metrics(base_dir, version):
    filepath = os.path.join(base_dir, f"version_{version}", "metrics.csv")
    df = pd.read_csv(filepath)
    cols = ['epoch', 'train_loss', 'train_mae', 'train_mse', 'val_loss', 'val_mae', 'val_mse']
    df = df[cols]
    df_merged = df.groupby('epoch', as_index=False).max()
    print(tabulate(df_merged, headers='keys', tablefmt='pretty', showindex=False))

def parse_test_args():
    parser = argparse.ArgumentParser(description="Test script for HCP model")
    parser.add_argument('--base_dir', type=str, required=True, 
                       help='Base directory containing version folders (e.g., /path/to/Result/high_dim_BCR/Mar17_GordonHCP)')
    parser.add_argument('--version', type=int, required=True, help='Version number of the training run to load')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size for prediction (overrides checkpoint value if provided)')
    return parser.parse_args()

def load_hparams(base_dir, version):
    hparams_path = os.path.join(base_dir, f"version_{version}", "hparams.yaml")
    with open(hparams_path, 'r') as f:
        hparams = yaml.safe_load(f)
        print(hparams)
    return argparse.Namespace(**hparams)

def main():
    torch.set_float32_matmul_precision("medium")
    args = parse_test_args()
    hparams = load_hparams(args.base_dir, args.version)
    process_and_print_metrics(args.base_dir, args.version)
    # Initialize data module with same parameters as training

    dm = GordonHCPDataModule(**vars(hparams))
    dm.setup(stage='fit')  # Setup both train and val splits
    
    # Load the best checkpoint based on minimum val_mae
    # checkpoint_dir = os.path.join(args.base_dir, f"version_{args.version}", "checkpoints")
    # checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    # checkpoint_path = min(checkpoints, key=lambda x: float(x.split('-val_mae=')[1].split('.ckpt')[0]))
    
    # print("Loading model from:", checkpoint_path)
    # model = LitORionModelOptimized.load_from_checkpoint(checkpoint_path)
    ckpt_path = hf_hub_download(repo_id="vhluong/MnemoDyn", filename="model.ckpt", revision="main")
    model = LitORionModelOptimized.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    
    base_output_dir = os.path.join(args.base_dir, f"version_{args.version}")
    train_output_dir = os.path.join(base_output_dir, "train_performance_ukb")
    val_output_dir = os.path.join(base_output_dir, "test_performance_ukb")
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)
    
    num_GPU = torch.cuda.device_count()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_GPU,
        precision="32",)
    
    train_bs = hparams.train_bs
    val_bs = hparams.valid_bs

    train_loader = DataLoader(dm.train_dataset, batch_size=train_bs, num_workers=10, drop_last=False)
    val_loader = DataLoader(dm.val_dataset, batch_size=val_bs, num_workers=10, drop_last=False)
    
    device = 'cuda'

    index = 0
    for p in model.parameters():
        p.requires_grad = False

    for x, coeffs, time in tqdm(train_loader, leave=False):
            
        x = x.to(device).float()
        coeffs    = coeffs.to(device).float()
        time      = time.to(device).float()
        
        with torch.no_grad():
            predicted = model(x, coeffs, time)
        np.save(os.path.join(train_output_dir, f"pred_batch_{index}.npy"), predicted.cpu().numpy())
        np.save(os.path.join(train_output_dir, f"orig_batch_{index}.npy"), x.cpu().numpy())        
        index += 1
  
    index = 0
    for x, coeffs, time in tqdm(val_loader, leave=False):

        x = x.to(device).float()
        coeffs    = coeffs.to(device).float()
        time      = time.to(device).float()
        with torch.no_grad():
            predicted = model(x, coeffs, time)
        np.save(os.path.join(val_output_dir, f"pred_batch_{index}.npy"), predicted.cpu().numpy())
        np.save(os.path.join(val_output_dir, f"orig_batch_{index}.npy"), x.cpu().numpy())
        index += 1
    print(f"Predictions saved in {train_output_dir} and {val_output_dir}")

if __name__ == "__main__":
    main()
