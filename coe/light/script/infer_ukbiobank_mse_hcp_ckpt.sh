export CUDA_VISIBLE_DEVICES=2
python infer_ukbiobank_mse.py \
  --ckpt /mnt/vhluong/Result/Orion_450/debug_GordonHCP/version_2/checkpoints/model-epoch=69-val_mae=0.0000.ckpt \
  --val_json ./norm_cache/norm_stats_seed34_normrobust_scaler_interpspline_28k_3rd.json \
  --filename timeseries.npy \
  --D 333 \
  --original_length 1200 \
  --batch_size 8 \
  --workers 8 \
  --normalize \
  --normalizer_kind robust_scaler \
  --lightning \
  --save_csv ukb_val_mse.csv
