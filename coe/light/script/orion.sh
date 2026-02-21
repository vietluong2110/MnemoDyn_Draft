export CUDA_VISIBLE_DEVICES=7

# Invoke your training script with all defaults from memory
python model/main.py \
  --train_bs 8 \
  --valid_bs 16 \
  --test_bs 16 \
  --dim_d 100 \
  --dim_k 300 \
  --lr 0.001 \
  --subset \
  --use_residual \
  --D 450 \
  --D_out 450 450 450 450\
  --K_LC 2 2 2 2 2 2 \
  --K_dense 4\
  --epoch 1000 \
  --h_rank 3 \
  --huber_delta 1.0 \
  --interpol spline \
  --loss_type total \
  --model_pred_save_freq 100 \
  --n_levels 5 \
  --nb 5 \
  --nonlinearity tanh \
  --normalize robust_scaler \
  --num_classes 1 \
  --num_sparse_LC 4 \
  --original_length 490 \
  --exp debug \
  --dataset_type GordonHCP \
  --storage /mnt/vhluong/Result/Orion_denoise/ \
  --seed 2025 \
  --wave db2 \