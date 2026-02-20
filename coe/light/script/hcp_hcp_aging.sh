#!/usr/bin/env bash
# run_train_gpu4.sh – launch training on CUDA device 4 with your default hyperparams

# Expose only physical GPU #4 (zero‐indexed)
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Invoke your training script with all defaults from memory
python main.py \
  --train_bs 8 \
  --valid_bs 16 \
  --test_bs 16 \
  --dim_d 150 \
  --dim_k 350 \
  --lr 0.001 \
  --subset \
  --use_residual \
  --D_out 333 333 333 333 \
  --K_LC 2 2 2 2 2 2 \
  --epoch 1000 \
  --h_rank 5 \
  --huber_delta 1.0 \
  --interpol spline \
  --loss_type total \
  --model_pred_save_freq 100 \
  --n_levels 6 \
  --nb 3 \
  --nonlinearity tanh \
  --normalize robust_scaler \
  --num_classes 1 \
  --num_sparse_LC 5 \
  --original_length 1200 \
  --exp debug \
  --dataset_type GordonHCP \
  --storage /mnt/sourav/Result/orion_333_pe/ \
  --seed 2025 \
  --wave db2 \
