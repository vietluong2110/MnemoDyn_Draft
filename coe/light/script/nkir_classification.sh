#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

declare -A versions=(
    # ["Orion_333"]=17
    # ["Orion_333_denoise"]=0
    # ["Orion_333_mask_autoencode_new"]=2
        # ["Orion_333_llm_config"]=0
    ["Orion_333_mask_autoencode_jepa"]=2

)

for seed in {2031..2034}; do
    for model in "${!versions[@]}"; do
        version=${versions[$model]}
        foundation_dir="/mnt/vhluong/Result/${model}/debug_GordonHCP/"

        echo "=== Running seed $seed | model $model | version $version ==="
        python nkir_sex_classification.py \
            --foundation-dir "$foundation_dir" \
            --train_bs 8 \
            --test_bs 4 \
            --lr 0.001 \
            --wd 0.0001 \
            --seed "$seed" \
            --version "$version"
    done
done

