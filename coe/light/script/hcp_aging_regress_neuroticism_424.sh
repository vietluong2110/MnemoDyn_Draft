export CUDA_VISIBLE_DEVICES=5
for seed in {2025..2034}; do
    echo "=== Running seed $seed ===" 
    python hcp_aging_regress_neuroticism_424_denoise.py \
                --foundation-dir "/mnt/vhluong/Result/Orion_424_llm_config/debug_GordonHCP/"  \
                --train_bs 1 \
                --test_bs  1 \
                --lr       0.001 \
                --wd       0.0001 \
                --seed "$seed" \
                --version 1
done