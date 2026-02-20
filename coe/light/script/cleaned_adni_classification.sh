export CUDA_VISIBLE_DEVICES=5
python cleaned_adni_classification.py \
          --foundation-dir "/mnt/sourav/Result/orion_333_pe_new_gradient_mapping/debug_GordonHCP/"  \
          --train_bs 16 \
          --test_bs  16 \
          --lr       0.001 \
          --wd       0.0001 \
          --seed     2024 \
          --version 2 \
