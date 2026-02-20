export CUDA_VISIBLE_DEVICES=6
for seed in {2025..2034}; do
    echo "=== Running seed $seed ===" 
    python hcp_aging_regress_flanker_424.py \
        --foundation-dir "/mnt/vhluong/Result/Orion_424_mask_autoencode_jepa_ukbiobank/debug_GordonHCP/"  \
        --train_bs 8 \
        --test_bs  4 \
        --lr       0.001 \
        --wd       0.0001 \
        --seed "$seed" \
        --version 9
done