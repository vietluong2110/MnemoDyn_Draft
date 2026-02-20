export CUDA_VISIBLE_DEVICES=7

python abide_classification_normal.py \
        --foundation-dir "/mnt/vhluong/Result/Orion_450_ukbiobank/debug_GordonHCP/"  \
        --train_bs 8 \
        --test_bs  4 \
        --lr       0.001 \
        --wd       0.0001 \
        --seed     2023 \
        --version 2