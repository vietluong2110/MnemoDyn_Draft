import os, sys, pdb, argparse, yaml
import pandas as pd
import numpy as np
from tabulate import tabulate
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from main_masked_autoencode import GordonHCPDataModule, LitORionModelOptimized  # Import from your main file
from tqdm import tqdm 
from hcp_dataset_2 import HCPA_Dataset, load_hcp_aging_parcellated_list, HCPADataModule

import torch.nn as nn

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
    # dm = GordonHCPDataModule(**vars(hparams))
    # dm.setup(stage='fit')  # Setup both train and val splits
    
    # Load the best checkpoint based on minimum val_mae
    checkpoint_dir = os.path.join(args.base_dir, f"version_{args.version}", "checkpoints")
    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    checkpoint_path = min(checkpoints, key=lambda x: float(x.split('-val_mae=')[1].split('.ckpt')[0]))
    print("Loading model from:", checkpoint_path)
    model = LitORionModelOptimized.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    base_output_dir = os.path.join(args.base_dir, f"version_{args.version}")
    train_output_dir = os.path.join(base_output_dir, "train_performance_hcp")
    val_output_dir = os.path.join(base_output_dir, "test_performance_hcp")
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)
    
    num_GPU = torch.cuda.device_count()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_GPU,
        precision="32",)
    
    train_bs = hparams.train_bs
    val_bs = hparams.valid_bs

    # train_loader = DataLoader(dm.train_dataset, batch_size=train_bs, num_workers=10, drop_last=False)
    # val_loader = DataLoader(dm.val_dataset, batch_size=val_bs, num_workers=10, drop_last=False)
    
    device = 'cuda'
  
    index = 0

    for p in model.parameters():
        p.requires_grad = False
    all_files, label_map = load_hcp_aging_parcellated_list(
        '/mnt/vhluong/HCPA_with_new_age_sex.csv',
        '/mnt/sourav/HCPAging/dtseries',
        label_column='interview_age',
    )
    
    hcpa_module = HCPADataModule(
        tsv_dir = '/mnt/sourav/HCPAging/dtseries', 
        file_list = all_files, 
        label_map = label_map, 
        duration = 1,
        original_length=490, 
        interpol='spline', 
        target_length=490, 
        num_parcels=450, 
        one_channel=-1,
        subset=True,  
        dim_D=450, 
        normalize='robust_scaler'
    )
    hcpa_module.setup()
    train_loader = hcpa_module.train_dataloader(train_bs)
    val_loader = hcpa_module.test_dataloader(1)

    
    device = 'cuda'

    # for x_obs_aug, coeffs, x_target, mask, time in tqdm(train_loader, leave=False):
        
    #     # expect: (x_obs_aug, coeffs, x_target, mask, time) — adapt if yours differs

    #     x_obs_aug = x_obs_aug.to(device).float()
    #     coeffs    = coeffs.to(device).float()
    #     time      = time.to(device).float()
    #     mask_t    = mask.to(device)           # bool or {0,1}
        
    #     with torch.no_grad():
    #         predicted = model(x_obs_aug, coeffs, time)
    #     np.save(os.path.join(train_output_dir, f"pred_batch_{index}.npy"), predicted.cpu().numpy())
    #     np.save(os.path.join(train_output_dir, f"orig_batch_{index}.npy"), x_target.cpu().numpy())
    #     np.save(os.path.join(train_output_dir, f"mask_batch_{index}.npy"), mask.cpu().numpy())
        
    #     index += 1
  
    # index = 0
    # for x_obs_aug, coeffs, x_target, mask, time in tqdm(val_loader, leave=False):
    #      # expect: (x_obs_aug, coeffs, x_target, mask, time) — adapt if yours differs

    #     x_obs_aug = x_obs_aug.to(device).float()
    #     coeffs    = coeffs.to(device).float()
    #     time      = time.to(device).float()
    #     mask_t    = mask.to(device)   
    #     with torch.no_grad():
    #         predicted = model(x_obs_aug, coeffs, time)
    #     np.save(os.path.join(val_output_dir, f"pred_batch_{index}.npy"), predicted.cpu().numpy())
    #     np.save(os.path.join(val_output_dir, f"orig_batch_{index}.npy"), x_target.cpu().numpy())
    #     np.save(os.path.join(val_output_dir, f"mask_batch_{index}.npy"), mask.cpu().numpy())
    #     index += 1
    # print(f"Predictions saved in {train_output_dir} and {val_output_dir}")

    index = 0
    for x, coeffs, y, time in tqdm(val_loader, leave=False):
         # expect: (x_obs_aug, coeffs, x_target, mask, time) — adapt if yours differs
  
        x = x.to(device).float()
        coeffs    = coeffs.to(device).float()
        time      = time.to(device).float()
        # mask_t    = mask.to(device)   
        with torch.no_grad():
            predicted = model(x, coeffs, time)
            loss = nn.MSELoss()(x, predicted)

            print(f"Loss {loss}")
        np.save(os.path.join(val_output_dir, f"pred_batch_{index}.npy"), predicted.cpu().numpy())
        np.save(os.path.join(val_output_dir, f"orig_batch_{index}.npy"), x.cpu().numpy())
        # np.save(os.path.join(val_output_dir, f"mask_batch_{index}.npy"), mask.cpu().numpy())
        index += 1
    print(f"Predictions saved in {train_output_dir} and {val_output_dir}")

if __name__ == "__main__":
    main()

