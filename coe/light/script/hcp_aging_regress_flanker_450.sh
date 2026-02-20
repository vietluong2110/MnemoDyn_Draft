export CUDA_VISIBLE_DEVICES=7
for seed in {2030..2030}; do
    echo "=== Running seed $seed ===" 
    python hcp_aging_regress_flanker_450.py \
    --foundation-dir "/mnt/vhluong/Result/Orion_denoise/debug_GordonHCP/" \
    --train_bs 8 \
    --test_bs  4 \
    --lr       0.001 \
    --wd       0.0001 \
    --seed "$seed" \
    --version 2
done