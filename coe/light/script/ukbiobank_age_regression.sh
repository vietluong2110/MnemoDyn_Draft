export CUDA_VISIBLE_DEVICES=2
for seed in {2025..2034}; do
    echo "=== Running seed $seed ===" 
    python ukbiobank_age_regression.py \
            --foundation-dir "/mnt/vhluong/Result/Orion_450_ukbiobank/debug_GordonHCP/"  \
            --train_bs 8 \
            --test_bs  4 \
            --lr       0.001 \
            --wd       0.0001 \
            --seed     "$seed" \
            --version 2
done