#!/bin/bash


for seed in {2031..2034}; do
    for model in "${!versions[@]}"; do
        version=${versions[$model]}
        foundation_dir="/mnt/vhluong/Result/${model}/debug_GordonHCP/"

        echo "=== Running seed $seed | model $model | version $version ==="
        python nkir_sex_classification_450.py \
            --foundation-dir "$foundation_dir" \
            --train_bs 8 \
            --test_bs 4 \
            --lr 0.001 \
            --wd 0.0001 \
            --seed "$seed" \
            --version "$version"
    done
done
