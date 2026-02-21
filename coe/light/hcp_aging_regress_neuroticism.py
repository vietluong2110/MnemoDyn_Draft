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

from scipy.stats import pearsonr

from plot_save import plot_and_save, manual_set_seed
from sklearn.preprocessing  import MinMaxScaler

import os, glob
from main import LitORionModelOptimized
# from hcp_aging_utils import load_bold_data_with_age
from sklearn.preprocessing import LabelEncoder
import logging
from sklearn.model_selection import train_test_split
from model.normalizer import Normalizer
from hcp_dataset import  HCPA_Dataset, load_hcp_aging_parcellated_list, HCPADataModule
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
	# # Configuration file path

	# arguments for dataset
	parser.add_argument('--seq_length', type=int, default=490, help='Total seqeunce length in time series')

	# Setting experiment arugments
	parser.add_argument("--seed", default=4741, type=int, help="Setting seed for the entire experiment")
	parser.add_argument("--exp", default='HCP_Aging_Regression_Neuroticism_450_ukbiobank_mask_jepa_braindynamo', help="Adjusted in code: Experiment foler name")

	# Setting model arguments
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


def main():
	print('here')
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args = parse_args()
	arg_dict = vars(args)

	exp_name = args.exp
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
	logger.info("Time step tensor shape: %s", tuple(time_step.shape))

	# save args dict
	with open(os.path.join(args.exp, 'arguments.pkl'), 'wb') as f:
		pickle.dump(arg_dict, f)

	# load pretrained model
	version_dir = os.path.join(args.foundation_dir, f"version_{args.version}")
	ckpts = glob.glob(os.path.join(version_dir, 'checkpoints', '*.ckpt'))
	assert ckpts, f"No checkpoints in {version_dir}/checkpoints"
	ckpt_path = max(ckpts, key=os.path.getctime)
	logger.info("Loading checkpoint %s", ckpt_path)

	lit = LitORionModelOptimized.load_from_checkpoint(ckpt_path, map_location=device)
	lit.eval()
	foundation = lit.model.to(device)
	# # 2. Load EMA weights
	# ema_state = torch.load(f"{version_dir}/ema_best_val_mse.pt", map_location=device)
	# foundation.load_state_dict(ema_state, strict=True)  # load EMA params
	for p in foundation.parameters():
		p.requires_grad = False
	
	time_step = time_step.to(device)
	all_files, label_map = load_hcp_aging_parcellated_list(
		'/mnt/vhluong/HCPA_with_neuro_scores.csv',
		'/mnt/sourav/HCPAging/dtseries',
		label_column='neo2_score_ne',
	)


	logger.info("Total files loaded: %d", len(all_files))

	train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

	logger.info("Train/Test split: %d/%d", len(train_files), len(test_files))
	train_dataset = HCPA_Dataset(
		data_dir='/mnt/sourav/HCPAging/dtseries',
		file_list=train_files,
		label_map=label_map,
		time_step=time_step,
		interpol=args.interpol,
		one_channel=None,
		subset=True,
		dim_D=450,
		target_length=args.seq_length,

		num_parcels=450,
	)
	logger.info("Train dataset size: %d", len(train_dataset))

	test_dataset = HCPA_Dataset(
		data_dir='/mnt/sourav/HCPAging/dtseries',
		file_list=test_files,
		label_map=label_map,
		time_step=time_step,
		interpol=args.interpol,
		one_channel=None,
		subset=True,
		dim_D=450,
		target_length=args.seq_length,
		num_parcels=450,
	)
	logger.info("Test dataset size: %d", len(test_dataset))
	normalizer = Normalizer(lit.hparams.normalize)
	normalizer.fit(train_dataset.data)
	train_dataset.data = normalizer.transform(train_dataset.data)
	test_dataset.data = normalizer.transform(test_dataset.data)
	logger.info("Data normalization (%s) applied", lit.hparams.normalize)

	train_dl = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True)
	test_dl = DataLoader(test_dataset, batch_size=args.test_bs, shuffle=False, drop_last = True)


	# all_files, label_map = load_hcp_file_list(
	# 	tsv_dir='/mnt/sourav/new_BOLD/BOLD_FC',
	# 	metadata_path='/mnt/sourav/new_BOLD/HCPA_metadata.csv',
	# 	label_column='Sex',
	# 	exclude_index=226,
	# 	max_files = None 
	# )
	# logger.info("Total files loaded: %d", len(all_files))
	# labels = list(label_map.values())
	# count_0 = sum(1 for v in labels if v == 0)
	# count_1 = sum(1 for v in labels if v == 1)

	# print(f"0s: {count_0}")
	# print(f"1s: {count_1}")

	# hcpa_module = HCPADataModule(
	# 	tsv_dir = '/mnt/sourav/new_BOLD/BOLD_FC', 
	# 	file_list = all_files, 
	# 	label_map = label_map, 
	# 	duration = args.duration,
	# 	original_length=args.seq_length, 
	# 	interpol=args.interpol, 
	# 	target_length=args.seq_length, 
	# 	num_parcels=333, 
	# 	one_channel=-1,
	# 	subset=True,  
	# 	dim_D=333, 
	# 	normalize=lit.hparams.normalize
	# )
	# hcpa_module.setup()

	# train_dl = hcpa_module.train_dataloader(args.train_bs)
	# test_dl = hcpa_module.test_dataloader(args.test_bs)

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
		nn.Linear(256, 1),
		nn.Flatten(start_dim=1),   # → [B, 10]
    	nn.Linear(490, 600),          # → [B, 1]
		nn.LayerNorm(600),
		nn.GELU(),
		nn.Dropout(0.1),

		nn.Linear(600, 300),          # → [B, 1]
		nn.LayerNorm(300),
		nn.GELU(),
		nn.Dropout(0.1),

		nn.Linear(300, 150),          # → [B, 1]
		nn.LayerNorm(150),
		nn.GELU(),
		nn.Dropout(0.1),

		nn.Linear(150, 75),          # → [B, 1]
		nn.LayerNorm(75),
		nn.GELU(),
		nn.Dropout(0.1),

		nn.Linear(75, 1)

	).to(device)
	

	pytorch_total_params = sum(p.numel() for p in regression_head.parameters() if p.requires_grad)
	print("Total number of trainable parameters: ", pytorch_total_params)


	recon_loss_fn = nn.MSELoss(reduction = 'mean')
	optimizer = torch.optim.AdamW(regression_head.parameters(), lr=args.lr, weight_decay=args.wd)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

	all_losses = {'train_total_loss': [], 'valid_total_loss': [], 'test_total_loss': []}
	result_dict = {'train_acc': [], 'test_acc': [],
				   'number_param': pytorch_total_params,
				   'train_time': [], 'memory': []}

	best_test_loss = float('inf')
	best_epoch = -1
	best_pearson = float('-inf')
	pearson_corr_list = []
	# ckpt = torch.load("/workspace/high_dim_BCR/coe/light/Result/HCP_Aging_Regression_Neuroticism/Results/21/model_0.pt", map_location="cpu")
	# state_dict = ckpt.get("state_dict", ckpt)  # handles both full‐model or state_dict
	# regression_head.load_state_dict(state_dict, strict=False)
	for epoch in range(args.epoch):
		logger.info("Starting epoch %d/%d", epoch+1, args.epoch)
		start_time = sys_time.time()
		start_mem = get_memory(device, reset=True)

		# training
		train_loss = 0.0
		for x, coeffs, y_true, time in tqdm(train_dl, leave=False):
			x, coeffs, y_true, time = (
				x.to(device).float(), coeffs.to(device).float(),
				y_true.to(device).float(), time.to(device).float()
			)
			with torch.no_grad():
				U = foundation(x, coeffs, time)
			features = U
			y_pred = regression_head(features)
			loss = ((y_pred.squeeze(1) - y_true) ** 2).mean()
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# range_sq = torch.tensor(scaler.data_range_ )      \
            #        .to(loss.device)                      \
            #        .type_as(loss)   # match dtype (float32/64)
			# loss_scaled = loss * range_sq 

			train_loss += loss.item()

		epoch_train_loss = train_loss / len(train_dl)
		train_time = sys_time.time() - start_time
		peak_mem = get_memory(device) - start_mem
		logger.info("Epoch %d TRAIN loss=%.4f  time=%.1fs  mem_peak=%.1fMB",
					epoch, epoch_train_loss, train_time, peak_mem)
		all_losses['train_total_loss'].append(epoch_train_loss)
		result_dict['train_time'].append(train_time)
		result_dict['memory'].append(peak_mem)

		# validation
		val_loss = 0.0
		all_preds = []
		all_targets = []
		logger.info("Epoch %d: running validation", epoch)
		regression_head.eval()
		with torch.no_grad():
			for x, coeffs, y_true, time in tqdm(test_dl, leave=False):
				x, coeffs, y_true, time = (
					x.to(device).float(), coeffs.to(device).float(),
					y_true.to(device).float(), time.to(device).float()
				)
				
				U = foundation(x, coeffs, time)
				features = U
				y_pred = regression_head(features)
				loss = ((y_pred.squeeze(1) - y_true) ** 2).mean()
				# range_sq = torch.tensor(scaler.data_range_ )      \
                #    .to(loss.device)                      \
                #    .type_as(loss)   # match dtype (float32/64)
				# loss_scaled = loss * range_sq 
				val_loss += loss.item()
				all_preds.append(y_pred.cpu().numpy())
				all_targets.append(y_true.cpu().numpy())
		epoch_val_loss = val_loss / len(test_dl)
		all_preds = np.concatenate(all_preds).squeeze()
		all_targets = np.concatenate(all_targets).squeeze()
		@torch.no_grad()
		def r2_score_torch_(y_true, y_pred):
			"""
			Compute R² using PyTorch. Accepts torch tensors or numpy arrays.
			"""
			# pdb.set_trace()
			# If numpy → convert to torch
			if isinstance(y_true, np.ndarray):
				y_true = torch.from_numpy(y_true)
			if isinstance(y_pred, np.ndarray):
				y_pred = torch.from_numpy(y_pred)

			# Detach (in case tensors require grad), cast to float
			y_true = y_true.detach().float()
			y_pred = y_pred.detach().float()

			# Optional: move y_true to same device as y_pred
			if y_true.device != y_pred.device:
				y_true = y_true.to(y_pred.device)

			# Flatten in case shapes are [B, 1] or [B, T, 1]
			y_true = y_true.view(-1)
			y_pred = y_pred.view(-1)

			ss_res = torch.sum((y_true - y_pred) ** 2)
			ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)

			if ss_tot == 0:
				# Put result on same device
				return torch.tensor(1.0, device=y_true.device) if ss_res == 0 else torch.tensor(0.0, device=y_true.device)

			return 1.0 - ss_res / ss_tot
		
		if len(np.unique(all_targets)) > 1:
			pearson_corr, _ = pearsonr(all_preds, all_targets)
		else:
			pearson_corr = float('nan')
		r2_score = r2_score_torch_(all_targets, all_preds)
		logger.info("Epoch %d VALID loss=%.4f, Pearson r=%.4f, R2 = %.4f", epoch, epoch_val_loss, pearson_corr, r2_score)
		all_losses['valid_total_loss'].append(epoch_val_loss)
		pearson_corr_list.append(pearson_corr)
		if epoch_val_loss < best_test_loss:
			best_test_loss = epoch_val_loss
			best_epoch = epoch

		scheduler.step(epoch_val_loss)
		plot_and_save(args.exp, all_losses, only_one=True)

		with open(os.path.join(args.exp, 'result.pkl'), 'wb') as f:
			pickle.dump(result_dict, f)

		if epoch % args.model_pred_save_freq == 0:
			torch.save(regression_head.state_dict(), os.path.join(args.exp, f'model_{epoch}.pt'))

	logger.info("Training complete. Best VALID loss=%.4f at epoch %d (Pearson r=%.4f)", best_test_loss, best_epoch, max(pearson_corr_list))
	torch.save(regression_head.state_dict(), os.path.join(args.exp, 'final_model.pt'))
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


	logger.info("Training summary saved to %s", summary_file)

	print("#####################################################")


# instantiate:


if __name__ == '__main__':
	main()