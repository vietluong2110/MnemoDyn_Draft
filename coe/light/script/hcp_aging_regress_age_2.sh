export CUDA_VISIBLE_DEVICES=0
for seed in {2030..2035}; do
    echo "=== Running seed $seed ===" 
    python hcp_aging_regress_2.py \
            --foundation-dir "/mnt/vhluong/Result/Orion_450_ukbiobank_36k/debug_GordonHCP"  \
            --train_bs 16 \
            --test_bs  16 \
            --lr       0.001 \
            --wd       0.0001 \
            --seed "$seed" \
            --version 1
done
