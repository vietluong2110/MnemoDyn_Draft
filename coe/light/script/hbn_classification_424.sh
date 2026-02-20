export CUDA_VISIBLE_DEVICES=6

declare -A versions=(
    ["Orion_424"]=2
    ["Orion_424_denoise"]=3
    ["Orion_424_mask_autoencode_new"]=1
    # ["Orion_424_llm_config"]=1
)

for seed in {2031..2034}; do
    for model in "${!versions[@]}"; do
        version=${versions[$model]}
        foundation_dir="/mnt/vhluong/Result/${model}/debug_GordonHCP/"

        echo "=== Running seed $seed | model $model | version $version ==="
        python hbn_classification_424.py \
          --foundation-dir "$foundation_dir"  \
          --train_bs 1 \
          --test_bs  1 \
          --lr       0.001 \
          --wd       0.0001 \
          --seed "$seed" \
          --version "$version"
    done
done