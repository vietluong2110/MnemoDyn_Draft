'''
Main file for classification task Eigenworm dataset, using BCR-DE model/ BCR-DE (noise) model
'''


import math
import os
import sys
import argparse
import random 
import numpy as np 
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

import pdb
import time as sys_time
import json
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from model.main import LitORionModelOptimized

from plot_save import plot_and_save, manual_set_seed

import os, glob
from model.normalizer import Normalizer
from abide_dataset import ABIDE_Dataset, load_abide_file_list
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


from collections import Counter


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description="Arguments for CDE in BCR")
    parser.add_argument("--foundation-dir", type=str, required=True, help='Path to your lightning version folder (e.g. robust_scaler..)')
    parser.add_argument("--version", type = int, required=True, help='which version subfolder to load (e.g. 0)')
    parser.add_argument('--normalize', default='minmax', type=str, choices=['standard', 'minmax', 'raw', 'robust_scaler', 'all_patient_all_voxel', 'per_patient_all_voxel',
                        'per_patient_per_voxel', 'per_voxel_all_patient', 'subtract_mean', 'subtract_mean_global_std', 'subtract_mean_99th_percentile'],
                        help='Normalization type: standard (zero mean, unit variance) or minmax (0-1 range)')
    parser.add_argument('--duration', default=1, type=float, help="Time duration of time series")
    # arguments for dataset
    parser.add_argument('--seq_length', type=int, default=490, help='Total seqeunce length in time series')
    parser.add_argument("--seed", default=4741, type=int, help="Setting seed for the entire experiment")
    parser.add_argument("--exp", default='ABIDE_classification_450_normal', help="Adjusted in code: Experiment foler name")
    parser.add_argument('--dim_D', default=7, type=int, help="Dimension of observable variable")
    parser.add_argument('--num_classes', default=2, type=int, help="Output dimensionality of regression head")
    parser.add_argument('--interpol', default='spline', type=str, help='Interpolation type to use')
    # training arguments
    parser.add_argument('--train_bs', default=64, type=int, help='Batchsize for train loader')
    parser.add_argument('--valid_bs', default=256, type=int, help='Batchsize for valid loader')
    parser.add_argument('--test_bs', default=256, type=int, help='Batchsize for test loader')
    parser.add_argument('--epoch', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--lr', default=0.01, type=float, help="Learning rate for the BCR_DE model")
    parser.add_argument('--wd', default=0.0001, type=float, help="Learning rate for the BCR_DE model")
    parser.add_argument('--model_pred_save_freq', default=10, type=int, help='Saving frequency of model prediction')
            
    args = parser.parse_args()
    return args

def get_memory(device, reset=False, in_mb=True):
    """ Gets the GPU usage. """
    if device is None:
        return float('nan')
    if device.type == 'cuda':
        if reset:
            torch.cuda.reset_max_memory_allocated(device)
        bytes = torch.cuda.max_memory_allocated(device)
        if in_mb:
            bytes = bytes / 1024 / 1024
        return bytes
    else:
        return float('nan')

def compute_acc(pred, target):
    probs = torch.softmax(pred, dim=1)
    winners = probs.argmax(dim=1)
    corrects = (winners == target)
    accuracy = corrects.sum().float() / float( target.size(0) )
    return accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    arg_dict = vars(args)

    exp_name   = args.exp
    base_result = os.path.join('Result', exp_name)
    if os.path.exists(base_result):
        nth_exp = len(os.listdir(os.path.join(base_result, 'Results'))) + 1
    else:
        nth_exp = 0
    args.exp = os.path.join(base_result, 'Results', str(nth_exp))
    os.makedirs(args.exp, exist_ok=True)

    # setup logging
    logger = setup_logging(args.exp)
    logger.info("Experiment directory: %s", args.exp)
    logger.info("Arguments: %s", arg_dict)

    # store args to CSV
    argument_storage_file = os.path.join(base_result, 'experiment.csv')
    if os.path.exists(argument_storage_file):
        with open(argument_storage_file, 'a') as csv_file:
            csv.DictWriter(csv_file, fieldnames=arg_dict.keys()).writerow(arg_dict)
    else:
        logger.info("First run: creating hyperparameter log %s", argument_storage_file)
        with open(argument_storage_file, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(arg_dict.keys())
            writer.writerow(arg_dict.values())

    manual_set_seed(args.seed)
    logger.info("Seed set to %d", args.seed)
    
    time_step = torch.from_numpy(np.arange(0, args.duration, args.duration / args.seq_length))
    time_step = time_step.to(device)
    logger.info("Time step tensor shape: %s", tuple(time_step.shape))

    # save args dict
    with open(os.path.join(args.exp, 'arguments.pkl'), 'wb') as f:
        pickle.dump(arg_dict, f)

    # load pretrained model (foundation)
    version_dir = os.path.join(args.foundation_dir, f"version_{args.version}")
    ckpts = glob.glob(os.path.join(version_dir, 'checkpoints', '*.ckpt'))
    assert ckpts, f"No checkpoints in {version_dir}/checkpoints"
    ckpt_path = max(ckpts, key=os.path.getctime)
    logger.info("Loading checkpoint %s", ckpt_path)

    lit = LitORionModelOptimized.load_from_checkpoint(ckpt_path, map_location=device)
    lit.eval()
    foundation = lit.model.to(device)
    for p in foundation.parameters():
        p.requires_grad = False
    # ─── Load file list & metadata CSV ───────────────────────────────────────────
    all_files, label_map_str = load_abide_file_list(
        '/mnt/sourav/ABIDE_dtseries/',
        '/mnt/vhluong/abide_paths_labels.csv',
        'label_num'
    )

    logger.info("Total files loaded: %d", len(all_files))

    labels = [label_map_str[f] for f in all_files]

    counts = Counter(labels)
    print(counts) 


    # ─── Split train/test ───────────────────────────────────────────────────────────
    train_files, test_files, train_labels, test_labels = train_test_split(all_files, labels, test_size=0.2, random_state=42)
    logger.info("Train/Test split: %d/%d", len(train_files), len(test_files))
    # ─── Instantiate ADNI_Dataset for classification (label_column='researchGroup') ─
    train_dataset = ABIDE_Dataset(
        data_dir       = '/mnt/sourav/ABIDE_dtseries/',
        file_list      = train_files,
        label_map      = label_map_str,         # now maps rel_path→(grp_idx, age)
        time_step      = time_step,             # pass in the tensor rather than a float
        interpol       = args.interpol,
        one_channel    = -1,
        subset         = True,                 # or True+dim_D if you want to subselect parcels
        dim_D          = 450,                  # must be a List[int] if subset=True
        target_length  = args.seq_length,       # e.g. 1200
        num_parcels    = 450,                   # make sure this matches whatever your .dtseries really has
        label_column   = 'dx'        # <-- this makes __getitem__ return grp_idx
    )
    logger.info("Train dataset size: %d", len(train_dataset))
    test_dataset = ABIDE_Dataset(
        data_dir       = '/mnt/sourav/ABIDE_dtseries/',
        file_list      = test_files,
        label_map      = label_map_str,
        time_step      = time_step,
        interpol       = args.interpol,
        one_channel    = -1,
        subset         = True,
        dim_D          = 450,
        target_length  = args.seq_length,
        num_parcels    = 450,
        label_column   = 'dx'
    )
    logger.info("Test dataset size: %d", len(test_dataset))
    # ─── Normalize (optional) ───────────────────────────────────────────────────────
    normalizer = Normalizer(lit.hparams.normalize)
    normalizer.fit(train_dataset.data)
    normalizer.transform(train_dataset.data)
    normalizer.transform(test_dataset.data)
    logger.info("Data normalization (%s) applied", lit.hparams.normalize)

    # ─── Create DataLoaders ─────────────────────────────────────────────────────────
    train_dl = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True)
    test_dl  = DataLoader(test_dataset, batch_size=args.test_bs, shuffle=False)
    # ≈962 561 total params
    regression_head = nn.Sequential(
        # 1) 333 → 512
        nn.Linear(lit.hparams.D, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(0.1),

        # 2) 512 → 1024
        nn.Linear(512, 1024),
        nn.LayerNorm(1024),
        nn.GELU(),
        nn.Dropout(0.1),

        # 3) 1024 → 256
        nn.Linear(1024, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(0.1),

        # final 256 → 1
        nn.Linear(256, 2)
    ).to(device)
    pytorch_total_params = sum(p.numel() for p in regression_head.parameters() if p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    recon_loss_fn = nn.MSELoss()
    class_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(regression_head.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    all_losses = {}
    all_losses['train_total_loss'] = []
    all_losses['valid_total_loss'] = []
    all_losses['test_total_loss'] = [] 

    result_dict = {}
    result_dict['train_acc'] = []
    result_dict['valid_acc'] = []
    result_dict['test_acc'] = []
    result_dict['train_f1'] = []
    result_dict['test_f1'] = []

    result_dict['number_param'] = pytorch_total_params
    result_dict['train_time'] = []
    result_dict['memory'] = []

    print("Start training")

    for epoch in range(args.epoch):
        epoch_total_loss = 0
        n_batches = 0
        correct = 0
        start_time = sys_time.time() 
        start_memory = get_memory(device, reset=True)
        train_trues = []
        train_preds = []
        for x, coeffs, y_true, time in tqdm(train_dl, leave=False):
            x, coeffs, y_true, time = x.to(device).float(), coeffs.to(device).float(), y_true.to(device), time.to(device).float()
            
            with torch.no_grad():
                U = foundation(x, coeffs, time)  # [B, T, dim_D_out]

            # features = torch.mean(U[:, 237, :], dim = 1) 
            features = U[:, -1, :]
            y_pred = regression_head(features)
            loss = class_loss_fn(y_pred, y_true) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()
            n_batches += 1

            pred = y_pred.argmax(dim=1, keepdim=True) 
            correct += pred.eq(y_true.view_as(pred)).sum().item()

            train_trues.extend(y_true.cpu().tolist())
            train_preds.extend(pred.cpu().tolist())

        result_dict['train_time'].append(sys_time.time()-start_time)
        result_dict['memory'].append(get_memory(device)-start_memory)
        train_f1  = f1_score(train_trues, train_preds, average='macro')
        print("Epoch: {}; Train: Total Loss:{}, Accuracy:{}, F1 {}".format(epoch
                                                              , epoch_total_loss/n_batches
                                                              , correct/len(train_dl.dataset)
                                                              , train_f1))
        
        all_losses['train_total_loss'].append(epoch_total_loss/n_batches)
        result_dict['train_acc'].append(correct/len(train_dl.dataset))

        
        epoch_total_loss = 0
        n_batches = 0
        correct = 0
        test_trues = []
        test_preds = []
        with torch.no_grad():
            for x, coeffs, y_true, time in tqdm(test_dl, leave=False):
                x, coeffs, y_true, time = x.to(device).float(), coeffs.to(device).float(), y_true.to(device), time.to(device).float()
                # x_pred, y_pred = model(x, coeffs, time_step)
                with torch.no_grad():
                    U = foundation(x, coeffs, time)  # [B, T, dim_D_out]
                
                # features = U[:, -1, :]                     # take final time-step
                # features = torch.mean(U, dim = 1)
                # y_pred = regression_head(features)         # [B, 1]
                # features = torch.mean(U[:, 237:242, :], dim = 1)  
                features = U[:, -1, :]
                y_pred = regression_head(features)
                
                loss = class_loss_fn(y_pred, y_true) 

                epoch_total_loss += loss.item()
                n_batches += 1

                pred = y_pred.argmax(dim=1, keepdim=True) 
                correct += pred.eq(y_true.view_as(pred)).sum().item()
                test_trues.extend(y_true.cpu().tolist())
                test_preds.extend(pred.cpu().tolist())

            test_f1 = f1_score(test_trues, test_preds, average='macro')
            if correct/len(test_dl.dataset) > 0.7:
                print("************************************")
                print("\t \t Test: Total Loss:{}, Accuracy:{},  F1: {}".format(
                        epoch_total_loss/n_batches, 
                        correct/len(test_dl.dataset),
                        test_f1))
                print("************************************")
            else:
                print("\t Test: Total Loss:{}, Accuracy:{}, F1: {}".format(
                    epoch_total_loss/n_batches,
                    correct/len(test_dl.dataset),
                    test_f1))
        
        all_losses['test_total_loss'].append(epoch_total_loss/n_batches)
        result_dict['test_acc'].append(correct/len(test_dl.dataset))
        result_dict['test_f1'].append(test_f1)
        val_loss = epoch_total_loss/n_batches
        scheduler.step(val_loss)
        
        plot_and_save(args.exp, all_losses, only_one=True)

        result_file = args.exp+'/result.pkl'
        with open(result_file, 'wb') as f:
            pickle.dump(result_dict, f)

        if epoch%args.model_pred_save_freq==0:
            torch.save(regression_head.state_dict(), args.exp+'/model_'+str(epoch)+'.pt')

    print("Best: Train{}, Test{}".format(max(result_dict['train_acc']), max(result_dict['test_acc'])))

    torch.save(regression_head.state_dict(), args.exp+'/final_model.pt')
    logger.info("Final model saved.")
    summary_file = os.path.join(args.exp, 'training_summary.csv')
    with open(summary_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # write header
        writer.writerow(['parameter', 'value'])
        # write all hparams
        writer.writerow(['train_bs', args.train_bs])
        writer.writerow(['test_bs', args.test_bs])
        writer.writerow(['lr', args.lr])
        writer.writerow(['wd', args.wd])
        writer.writerow(['seed', args.seed])

        # write best metrics
        writer.writerow(['best_accuracy', max(result_dict['test_acc'])])
    logger.info("Training summary saved to %s", summary_file)

    print("#####################################################")

if __name__ == '__main__':
    main()