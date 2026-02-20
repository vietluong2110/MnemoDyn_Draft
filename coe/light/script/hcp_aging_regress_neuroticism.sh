export CUDA_VISIBLE_DEVICES=1
for seed in {2031..2034}; do
    echo "=== Running seed $seed ===" 
    python hcp_aging_regress_neuroticism.py \
            --foundation-dir "/mnt/vhluong/Result/Orion_333_llm_config/debug_GordonHCP/"  \
            --train_bs 1 \
            --test_bs  1 \
            --lr       0.001 \
            --wd       0.0001 \
            --seed "$seed" \
            --version 0
done 
