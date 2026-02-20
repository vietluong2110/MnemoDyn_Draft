export CUDA_VISIBLE_DEVICES=4
for seed in {2026..2034}; do
  echo "=== Running seed $seed ===" 
  python hcp_aging_regress_424.py \
        --foundation-dir "/mnt/vhluong/Result/Orion_424/debug_GordonHCP/"  \
        --train_bs 16 \
        --test_bs  16 \
        --lr       0.001 \
        --wd       0.0001 \
        --seed "$seed" \
        --version 2 
done
