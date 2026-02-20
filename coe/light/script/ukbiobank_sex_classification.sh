export CUDA_VISIBLE_DEVICES=1
for seed in {2029..2034}; do
    echo "=== Running seed $seed ===" 
    python ukbiobank_sex_classification.py \
            --foundation-dir "/mnt/vhluong/Result/Orion_450_ukbiobank_36k/debug_GordonHCP/"  \
            --train_bs 8 \
            --test_bs  4 \
            --lr       0.001 \
            --wd       0.0001 \
            --seed     "$seed" \
            --version 2
done