export CUDA_VISIBLE_DEVICES=6
for seed in 2033; do
    echo "=== Running seed $seed ===" 
    python abide_classification_normal.py \
            --foundation-dir "/mnt/vhluong/Result/Orion_450_ukbiobank/debug_GordonHCP/"  \
            --train_bs 8 \
            --test_bs  4 \
            --lr       0.001 \
            --wd       0.0001 \
            --seed     "$seed" \
            --version 2
done