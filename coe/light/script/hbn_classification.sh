export CUDA_VISIBLE_DEVICES=7

declare -A versions=(
    ["Orion_333"]=17
    # ["Orion_333_denoise"]=0
    # ["Orion_333_mask_autoencode_new"]=2
    ["Orion_333_llm_config"]=0
)

for seed in {2031..2034}; do
    for model in "${!versions[@]}"; do
        version=${versions[$model]}
        foundation_dir="/mnt/vhluong/Result/${model}/debug_GordonHCP/"

        echo "=== Running seed $seed | model $model | version $version ==="
        python hbn_classification.py \
          --foundation-dir "$foundation_dir"  \
          --train_bs 1 \
          --test_bs  1 \
          --lr       0.001 \
          --wd       0.0001 \
          --seed "$seed" \
          --version "$version"
    done
done
