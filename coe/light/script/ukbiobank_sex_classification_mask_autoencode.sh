export CUDA_VISIBLE_DEVICES=0
for seed in {2030..2034}; do
    echo "=== Running seed $seed ===" 
    python ukbiobank_sex_classification.py \
            --foundation-dir "/mnt/vhluong/Result/Orion_450_mask_autoencode_ukbiobank_full_dataset_30_5_split_patch_length_30_loss_all/debug_GordonHCP/"  \
            --train_bs 8 \
            --test_bs  4 \
            --lr       0.001 \
            --wd       0.0001 \
            --seed     "$seed" \
            --exp "UKBiobank_Sex_classification_450_mask_autoencode"\
            --version 6
done