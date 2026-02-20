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

import os, glob
from main import LitORionModelOptimized
# from hcp_aging_utils import load_bold_data_with_age
from sklearn.preprocessing import LabelEncoder
import logging
from sklearn.preprocessing  import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from normalizer import Normalizer_classification
from hcp_dataset_450 import HCPA_Dataset, load_hcp_aging_parcellated_list, HCPADataModule
def parse_args():
	'''
	Parse input arguments
	'''
	parser = argparse.ArgumentParser(description="Arguments for CDE in BCR")
	parser.add_argument("--foundation-dir", type=str, required=True, help='Path to your lightning version folder (e.g. robust_scaler..)')
	parser.add_argument("--version", type = int, required=True, help='which version subfolder to load (e.g. 0)')
	# parser.add_argument('--one_channel', default=1, type=int, help="Use only this dimension")
	# parser.add_argument('--subset', action='store_true', default=False, help="Use first dim_D channel only")
	parser.add_argument('--normalize', default='minmax', type=str, choices=['standard', 'minmax', 'raw', 'robust_scaler', 'all_patient_all_voxel', 'per_patient_all_voxel',
						'per_patient_per_voxel', 'per_voxel_all_patient', 'subtract_mean', 'subtract_mean_global_std', 'subtract_mean_99th_percentile'],
						help='Normalization type: standard (zero mean, unit variance) or minmax (0-1 range)')
	parser.add_argument('--duration', default=1, type=float, help="Time duration of time series")


	# # Configuration file path
	# parser.add_argument('--config_path', type=str, default='./configs/classification/eigenworm.json', help='Path to data configuration file')

	# arguments for dataset
	parser.add_argument('--seq_length', type=int, default=490, help='Total seqeunce length in time series')
	# parser.add_argument('--data_path', type=str, default='../../../data/', help='Path to data directory for BIDMC32 and Eigenworm dataset. Check README')

	# Setting experiment arugments
	parser.add_argument("--seed", default=4741, type=int, help="Setting seed for the entire experiment")
	parser.add_argument("--exp", default='HCP_Aging_Regression_450_Few_Shot_Model', help="Adjusted in code: Experiment foler name")

	# Setting model arguments
	# parser.add_argument('--noise_model', default=False, action='store_true', help='Whether to use noise regularized model')
	# parser.add_argument('--wave', default='db2', type=str, help='Type of pywavelet')
	parser.add_argument('--dim_D', default=7, type=int, help="Dimension of observable variable")
	# parser.add_argument('--dim_D_out', default=1, type=int, help="Dimension of predicte variable")
	# parser.add_argument('--dim_d', default=3, type=int, help="Latent dimension of evolution")
	# parser.add_argument('--dim_k', default=3, type=int, help="Dimension of h_theta (first)")
	parser.add_argument('--num_classes', default=2, type=int, help="Output dimensionality of regression head")
	# parser.add_argument('--nonlinearity', default='tanh', type=str, help='Non lienarity to use')
	# parser.add_argument('--n_levels', default=8, type=int, help="numberr of levels of wavelet decomposition")
	# parser.add_argument('--K_dense', default=1, type=int, help="Number of dense layers")
	# parser.add_argument('--K_LC', default=4, type=int, help="Number of LC layers per level")
	# parser.add_argument('--nb', default=3, type=int, help="Diagonal banded length, dertimine the kernel size")
	# parser.add_argument('--num_sparse_LC', default=6, type=int, help="Number of sparse LC unit")
	parser.add_argument('--interpol', default='spline', type=str, help='Interpolation type to use')
	# parser.add_argument('--use_cheap_sparse_LC', default=True, action='store_false', help='Whether to use sparse Locally connected layer')

	# training arguments
	parser.add_argument('--train_bs', default=64, type=int, help='Batchsize for train loader')
	parser.add_argument('--valid_bs', default=256, type=int, help='Batchsize for valid loader')
	parser.add_argument('--test_bs', default=256, type=int, help='Batchsize for test loader')
	parser.add_argument('--epoch', default=150, type=int, help='Number of epochs to train')
	parser.add_argument('--lr', default=0.01, type=float, help="Learning rate for the BCR_DE model")
	parser.add_argument('--wd', default=0.0001, type=float, help="Learning rate for the BCR_DE model")
	parser.add_argument('--model_pred_save_freq', default=10, type=int, help='Saving frequency of model prediction')
	parser.add_argument('--time_step', default=-1, type = int, help = 'time step')
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
	for p in foundation.parameters():
		p.requires_grad = False
	
	time_step = time_step.to(device)
	all_files, label_map = load_hcp_aging_parcellated_list(
		'/mnt/vhluong/HCPA_with_new_age_sex.csv',
		'/mnt/sourav/HCPAging/dtseries',
		label_column='interview_age',
	)
	logger.info("Total files loaded: %d", len(all_files))
	train_files, test_files = train_test_split(all_files, test_size=0.3, random_state=42)

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

	normalizer = Normalizer_classification(lit.hparams.normalize)
	normalizer.fit(train_dataset.data)
	train_dataset.data = normalizer.transform(train_dataset.data)
	test_dataset.data = normalizer.transform(test_dataset.data)
	logger.info("Data normalization (%s) applied", lit.hparams.normalize)

	train_dl = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True)
	test_dl = DataLoader(test_dataset, batch_size=args.test_bs, shuffle=False, drop_last = True)

	regression_head = nn.Sequential(
		# 1) 333 → 512
		nn.Linear(450, 512),     
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
	# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)
	# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    # optimizer,
    # mode='min',
    # factor=0.5,         # drop by ½ instead of ¼
    # patience=3,         # wait only 3 epochs
    # threshold=1e-4,     # what “improvement” means
    # verbose=True
	# )

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.epoch,     # or total number of iterations
    eta_min=1e-6          # minimum LR at the end of schedule
	)
	all_losses = {'train_total_loss': [], 'valid_total_loss': [], 'test_total_loss': []}
	result_dict = {'train_acc': [], 'test_acc': [],
				   'number_param': pytorch_total_params,
				   'train_time': [], 'memory': []}

	best_test_loss = float('inf')
	best_epoch = -1
	best_pearson = float('-inf')

	for epoch in range(args.epoch):
		logger.info("Starting epoch %d/%d", epoch+1, args.epoch)
		start_time = sys_time.time()
		start_mem = get_memory(device, reset=True)

		# training
		train_loss = 0.0
		regression_head.train()
		for x, coeffs, y_true, time in tqdm(train_dl, leave=False):
			save_dir = "reconstruction_outputs_hcp"
			os.makedirs(save_dir, exist_ok=True)
			x, coeffs, y_true, time = (
				x.to(device).float(), coeffs.to(device).float(),
				y_true.to(device).float(), time.to(device).float()
			)
			with torch.no_grad():
				U = foundation(x, coeffs, time)
			x_np = x.cpu().numpy()
			U_np = U.cpu().numpy()

			np.save(f"{save_dir}/pred_batch_0.npy",  x_np)
			np.save(f"{save_dir}/orig_batch_0.npy",  U_np)
			# pdb.set_trace()
			# U = U.transpose(1, 2)
			# U = torch.transpose(U[5], 1, 2)
			# features = U[:, -10: , :]
			features = U
			## last 10 time step, last time step 
			# features = features.unsqueeze(1)
			y_pred = regression_head(features)
			# y_pred = y_pred.mean(dim = 1)
			loss = ((y_true - y_pred.squeeze(1)) ** 2).mean()
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

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
				# U = torch.transpose(U[5], 1, 2)
				# features = U[:, -10: , :]
				features = U
				# features = features.unsqueeze(1)
				y_pred = regression_head(features)
				# y_pred = y_pred.mean(dim = 1)
				loss = ((y_true - y_pred.squeeze(1)) ** 2).mean()
				val_loss += loss.item()
				all_preds.append(y_pred.cpu().numpy())
				all_targets.append(y_true.cpu().numpy())
		# pdb.set_trace()
		epoch_val_loss = val_loss / len(test_dl)
		all_preds = np.concatenate(all_preds).squeeze()
		all_targets = np.concatenate(all_targets).squeeze()
		# pdb.set_trace()
		if len(np.unique(all_targets)) > 1:
			pearson_corr, _ = pearsonr(all_preds, all_targets)
		else:
			pearson_corr = float('nan')
		print(pearson_corr)
		logger.info("Epoch %d VALID loss=%.4f, Pearson r=%.4f", epoch, epoch_val_loss, pearson_corr)
		all_losses['valid_total_loss'].append(epoch_val_loss)

		if epoch_val_loss < best_test_loss:
			best_test_loss = epoch_val_loss
			best_epoch = epoch
		if best_pearson < pearson_corr:
			best_pearson = pearson_corr

		scheduler.step()
		plot_and_save(args.exp, all_losses, only_one=True)

		with open(os.path.join(args.exp, 'result.pkl'), 'wb') as f:
			pickle.dump(result_dict, f)

		if epoch % args.model_pred_save_freq == 0:
			torch.save(regression_head.state_dict(), os.path.join(args.exp, f'model_{epoch}.pt'))

	logger.info("Training complete. Best VALID loss=%.4f at epoch %d (Pearson r=%.4f)", best_test_loss, best_epoch, best_pearson)
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


if __name__ == '__main__':
	main()
