'''
Thie file is for all sharable utility function
'''
import torch 
import numpy as np 
import os, sys
import pdb
import pickle
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd

def save_data(true, pred, epoch, data_type, exp_dir):
	dirname = exp_dir+'/model_prediction/'
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	fname = dirname+data_type+'_'+str(epoch)
	torch.save(true, fname+'_true.pt')
	torch.save(pred, fname+'_pred.pt')
	return

def save_data_couple(true, pred, ip_se, epoch, data_type, exp_dir):
	dirname = exp_dir+'/model_prediction/'
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	fname = dirname+data_type+'_'+str(epoch)
	torch.save(true, fname+'_true.pt')
	torch.save(pred, fname+'_pred.pt')
	torch.save(ip_se, fname+'_ip_se.pt')
	return

def save_mask_data(masked, full, pred, mask_indices, epoch, data_type, exp_dir):
	dirname = exp_dir+'/model_prediction/'
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	fname = dirname+data_type+'_'+str(epoch)
	torch.save(masked, fname+'_masked.pt')
	torch.save(full, fname+'_full.pt')
	torch.save(pred, fname+'_pred.pt')
	torch.save(mask_indices, fname+'_mask_indices.pt')
	return

def save_fixed_mask_data(masked, full, pred, epoch, data_type, exp_dir):
	dirname = exp_dir+'/model_prediction/'
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	fname = dirname+data_type+'_'+str(epoch)
	torch.save(masked, fname+'_masked.pt')
	torch.save(full, fname+'_full.pt')
	torch.save(pred, fname+'_pred.pt')
	return

def plot_and_save(exp, all_losses, only_one=False):
	loss_file = exp+'/all_losses.pkl'
	with open(loss_file, 'wb') as f:
		pickle.dump(all_losses, f)

	plt.plot(all_losses['train_total_loss'], label='Training')
	plt.plot(all_losses['valid_total_loss'], label='Validation')
	plt.plot(all_losses['test_total_loss'], label='Test')
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Total loss")
	plt.savefig(exp+'/TotalLoss.png')
	plt.cla()
	plt.clf()
	plt.close()

	if only_one:
		return

	plt.plot(all_losses['train_recon_loss'], label='Training')
	plt.plot(all_losses['valid_recon_loss'], label='Validation')
	plt.plot(all_losses['test_recon_loss'], label='Test')
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Reconstruction loss")
	plt.savefig(exp+'/ReconstructionLoss.png')
	plt.cla()
	plt.clf()
	plt.close()

	plt.plot(all_losses['train_pred_loss'], label='Training')
	plt.plot(all_losses['valid_pred_loss'], label='Validation')
	plt.plot(all_losses['test_pred_loss'], label='Test')
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Prediction loss")
	plt.savefig(exp+'/PredictionLoss.png')
	plt.cla()
	plt.clf()
	plt.close()
	return

def plot_and_save_l1(exp, all_losses, only_one=False):
	loss_file = exp+'/all_losses.pkl'
	with open(loss_file, 'wb') as f:
		pickle.dump(all_losses, f)

	plt.plot(all_losses['train_total_loss'], label='Training')
	plt.plot(all_losses['valid_total_loss'], label='Validation')
	plt.plot(all_losses['test_total_loss'], label='Test')
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Total loss")
	plt.savefig(exp+'/TotalLoss.png')
	plt.cla()
	plt.clf()
	plt.close()

	if only_one:
		return

	plt.plot(all_losses['train_recon_loss'], label='Training')
	plt.plot(all_losses['valid_recon_loss'], label='Validation')
	plt.plot(all_losses['test_recon_loss'], label='Test')
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Reconstruction loss")
	plt.savefig(exp+'/ReconstructionLoss.png')
	plt.cla()
	plt.clf()
	plt.close()

	plt.plot(all_losses['train_l1_loss'], label='Training')
	plt.plot(all_losses['valid_l1_loss'], label='Validation')
	plt.plot(all_losses['test_l1_loss'], label='Test')
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Prediction loss")
	plt.savefig(exp+'/PredictionLoss.png')
	plt.cla()
	plt.clf()
	plt.close()
	return

def plot_and_save_L1_TV(exp, all_losses):
	loss_file = exp+'/all_losses.pkl'
	with open(loss_file, 'wb') as f:
		pickle.dump(all_losses, f)

	plt.plot(all_losses['train_recon_loss'], label='Training')
	plt.plot(all_losses['valid_recon_loss'], label='Validation')
	plt.plot(all_losses['test_recon_loss'], label='Test')
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Reconstruction loss")
	plt.savefig(exp+'/ReconLoss.png')
	plt.cla()
	plt.clf()
	plt.close()

	plt.plot(all_losses['train_L1_loss'], label='Training')
	plt.plot(all_losses['valid_L1_loss'], label='Validation')
	plt.plot(all_losses['test_L1_loss'], label='Test')
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("L1 loss")
	plt.savefig(exp+'/L1Loss.png')
	plt.cla()
	plt.clf()
	plt.close()

	plt.plot(all_losses['train_TV_loss'], label='Training')
	plt.plot(all_losses['valid_TV_loss'], label='Validation')
	plt.plot(all_losses['test_TV_loss'], label='Test')
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("TV loss")
	plt.savefig(exp+'/TVLoss.png')
	plt.cla()
	plt.clf()
	plt.close()
	return

def manual_set_seed(seed):
	print("Setting all seeds to: ", seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def self_save_file():
	train_x = np.load('../data/se2se/RR/train_seq.npy')
	val_x = np.load('../data/se2se/RR/val_seq.npy')
	test_x = np.load('../data/se2se/RR/test_seq.npy')
	train_y = np.load('../data/se2se/RR/train_pred.npy')
	val_y = np.load('../data/se2se/RR/val_pred.npy')
	test_y = np.load('../data/se2se/RR/test_pred.npy')
	return train_x, val_x, test_x, train_y, val_y, test_y