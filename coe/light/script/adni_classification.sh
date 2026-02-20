export CUDA_VISIBLE_DEVICES=7
# for seed in {2025..2030}; do
#         echo "=== Running seed $seed"
#         python adni_classification_oversampling.py \
#                 --foundation-dir "/mnt/vhluong/Result/Orion_450_ukbiobank_full_dataset_30_5_split/debug_GordonHCP/"  \
#                 --train_bs 8 \
#                 --test_bs  4 \
#                 --lr       0.001 \
#                 --wd       0.0001 \
#                 --seed     "$seed" \
#                 --version 1

# done

python adni_classification_oversampling.py \
                --foundation-dir "/mnt/vhluong/Result/Orion_450_denoise/debug_GordonHCP/"  \
                --train_bs 8 \
                --test_bs  4 \
                --lr       0.001 \
                --wd       0.0001 \
                --seed     2025 \
                --version 3
