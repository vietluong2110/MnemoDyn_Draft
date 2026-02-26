export CUDA_VISIBLE_DEVICES=7
  python hcp_aging_regress.py \
        --foundation-dir "/mnt/vhluong/Result/Orion_450_ukbiobank/debug_GordonHCP/"   \
        --train_bs 16 \
        --test_bs  16 \
        --lr       0.001 \
        --wd       0.0001 \
        --seed 2030 \
        --version 2
