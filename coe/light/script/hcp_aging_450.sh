export CUDA_VISIBLE_DEVICES=7
for seed in {2026..2026}; do
  echo "=== Running seed $seed ===" 
  python hcp_aging_regress_450.py \
        --foundation-dir "/mnt/vhluong/Result/Orion_50_percent_dataset/debug_GordonHCP/"  \
        --train_bs 16 \
        --test_bs  16 \
        --lr       0.001 \
        --wd       0.0001 \
        --seed "$seed" \
        --version 1
done
