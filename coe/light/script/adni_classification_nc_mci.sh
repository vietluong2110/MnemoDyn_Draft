export CUDA_VISIBLE_DEVICES=7
python adni_classification_sex.py \
                --foundation-dir "/mnt/vhluong/Result/Orion_450_denoise/debug_GordonHCP/"  \
                --train_bs 8 \
                --test_bs  4 \
                --lr       0.001 \
                --wd       0.0001 \
                --seed     2025 \
                --version 3
