export CUDA_VISIBLE_DEVICES=2

# Invoke your training script with all defaults from memory
python main_masked_autoencode.py \
  --train_bs 8 \
  --valid_bs 16 \
  --test_bs 16 \
  --dim_d 100 \
  --dim_k 300 \
  --lr 0.001 \
  --subset \
  --use_residual \
  --D  450 \
  --D_out 450 450 450\
  --K_LC 2 2 2 2 2 2 \
  --K_dense 4\
  --epoch 1000 \
  --h_rank 3 \
  --huber_delta 1.0 \
  --interpol spline \
  --loss_type total \
  --model_pred_save_freq 100 \
  --n_levels 4 \
  --nb 5 \
  --nonlinearity tanh \
  --normalize robust_scaler \
  --num_classes 1 \
  --num_sparse_LC 4 \
  --original_length 490 \
  --exp debug \
  --dataset_type GordonHCP \
  --storage /mnt/vhluong/Result/Orion_450_mask_autoencode_ukbiobank_full_dataset_30_5_split_patch_length_30_loss_all/ \
  --seed 2025 \
  --wave db2 \
  