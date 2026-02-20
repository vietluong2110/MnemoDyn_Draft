#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
#!/usr/bin/env bash
set -euo pipefail

# Root directory containing all the *_GordonHCP folders
BASE_PATH="/home/vhluong/media/vhluong_nas/Result/high_dim_BCR"

# You can either enumerate them manually...
FOLDERS=(
#   all_patient_all_voxel_GordonHCP
  minmax_GordonHCP
  per_patient_all_voxel_GordonHCP
  per_patient_per_voxel_GordonHCP
  per_voxel_all_patient_GordonHCP
  raw_GordonHCP
  robust_scaler_GordonHCP
  standard_GordonHCP
  subtract_mean_99th_percentile_GordonHCP
  subtract_mean_global_std_GordonHCP
  subtract_mean_GordonHCP
)

# …or simply glob for them:
# mapfile -t FOLDERS < <(find "$BASE_PATH" -maxdepth 1 -type d -name "*_GordonHCP" -printf "%f\n")

for F in "${FOLDERS[@]}"; do
  DIR="${BASE_PATH}/${F}"
  if [[ ! -d "$DIR" ]]; then
    echo "⚠️  Skipping missing directory: $DIR"
    continue
  fi

  # find the latest version_* folder, sort by version number, pick the last
  latest_dir=$(find "$DIR" -maxdepth 1 -type d -name 'version_*' \
               | sort -V \
               | tail -n1)

  if [[ -z "$latest_dir" ]]; then
    echo "⚠️  No version_* subfolders found in $DIR, skipping."
    continue
  fi

  # strip off “version_” to get just the number
  version=$(basename "$latest_dir" | cut -d'_' -f2)

  echo ""
  echo "=== Processing $DIR  →  latest version = $version  ==="
  python light_test.py --base_dir "$DIR" --version "$version"
done