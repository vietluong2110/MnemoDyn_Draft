export CUDA_VISIBLE_DEVICES=2
for seed in {2026..2034}; do
  echo "=== Running seed $seed ==="
python hbn_regression.py \
          --foundation-dir "/mnt/vhluong/Result/Orion_450_ukbiobank_36k/debug_GordonHCP/"  \
          --train_bs 8 \
          --test_bs  4 \
          --lr       0.001 \
          --wd       0.0001 \
          --seed     "$seed" \
       --version 2 
done
