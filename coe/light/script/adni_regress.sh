#!/usr/bin/env bash
set -euo pipefail

GPUS=(1)
CHECKPOINTS=(18)
FOUNDATION_DIR="/mnt/sourav/Result/Orion_333/debug_GordonHCP"
SCRIPT="cleaned_adni_regression.py"

# ─── 1) Define your search space ───────────────────────────────────────────────
train_bs_list=(16)
test_bs_list=(16)
lr_list=(0.001 0.01 0.1)
wd_list=(0.0001 0.001 0.01)
seed_list=(2023 2024 2025)

combos=()
for train_bs in "${train_bs_list[@]}";  do
for test_bs  in "${test_bs_list[@]}";   do
for lr       in "${lr_list[@]}";        do
for wd       in "${wd_list[@]}";        do
for seed    in "${seed_list[@]}";     do
  # make a short name to both:
  #   • create an output folder under FOUNDATION_DIR
  #   • let us check “if it exists → skip”
  name="bs${train_bs}_tbs${test_bs}_lr${lr}_wd${wd}_seed${seed}"
  combos+=("$name:$train_bs:$test_bs:$lr:$wd:$seed")
done
done
done
done
done


for entry in "${combos[@]}"; do
  IFS=":" read name train_bs test_bs lr wd seed <<< "$entry"
  # outdir="$FOUNDATION_DIR/$name"

  # # skip if output dir already exists
  # if [ -d "$outdir" ]; then
  #   echo "⏭️  Skipping $name (already done)"
  #   continue
  # fi

  echo "🔖 Scheduling $name…"
  scheduled=false
  while [ "$scheduled" = false ]; do
    for gpu in "${GPUS[@]}"; do
      # is this GPU “occupied”?
      if [ -n "${busy[$gpu]:-}" ]; then
        continue
      fi

      echo "🚀 Launching $name on GPU $gpu"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        python "$SCRIPT" \
          --foundation-dir "$FOUNDATION_DIR" \
          --train_bs "$train_bs" \
          --test_bs  "$test_bs" \
          --lr       "$lr" \
          --wd       "$wd" \
          --seed     "$seed" \
          --version 17
        # when it finishes, mark GPU free again
        unset busy[$gpu]
      ) &

      busy[$gpu]=1
      scheduled=true
      break
    done

    if [ "$scheduled" = false ]; then
      echo "⏳ All GPUs busy, retrying in 30s…"
      sleep 30
    fi
  done
done

wait
echo "✅ All hyper‐parameter runs complete!"