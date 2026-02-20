export CUDA_VISIBLE_DEVICES=6,7

# Invoke your training script with all defaults from memory
python main_denoise.py \
  --train_bs 4 \
  --valid_bs 4 \
  --test_bs 16 \
  --dim_d 100 \
  --dim_k 350 \
  --lr 0.001 \
  --subset \
  --use_residual \
  --D 424 \
  --D_out 424 424 424 424 \
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
  --original_length 1200 \
  --exp debug \
  --dataset_type GordonHCP \
  --storage /mnt/vhluong/Result/Orion_424_denoise/ \
  --seed 2025 \
  --wave db2 \