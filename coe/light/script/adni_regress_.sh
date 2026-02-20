export CUDA_VISIBLE_DEVICES=2
python cleaned_adni_regression.py \
          --foundation-dir "/mnt/sourav/Result/orion_333_pe_new_gradient_mapping/debug_GordonHCP/version_2"  \
          --train_bs 16 \
          --test_bs  16 \
          --lr       0.001 \
          --wd       0.0001 \
          --seed     2025 \
          --version 17 \
          --label_column 'subjectAge'