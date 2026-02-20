export CUDA_VISIBLE_DEVICES=6
python adhd_200_sex_classification.py \
          --foundation-dir "/mnt/vhluong/Result/Orion_424/debug_GordonHCP/"  \
          --train_bs 8 \
          --test_bs  4 \
          --lr       0.001 \
          --wd       0.0001 \
          --seed     31 \
          --version 2 \