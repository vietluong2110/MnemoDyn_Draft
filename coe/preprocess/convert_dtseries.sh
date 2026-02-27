#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="derivatives/preprocessed"
CIFTI_DIR="dtseries"

# fs_LR 32k surfaces + atlas/template
TEMPLATE_DIR=".."
TEMPLATE_L_MID="${TEMPLATE_DIR}/fs_LR.32k.L.midthickness.surf.gii"
TEMPLATE_L_WHITE="${TEMPLATE_DIR}/fs_LR.32k.L.white.surf.gii"
TEMPLATE_L_PIAL="${TEMPLATE_DIR}/fs_LR.32k.L.pial.surf.gii"
TEMPLATE_R_MID="${TEMPLATE_DIR}/fs_LR.32k.R.midthickness.surf.gii"
TEMPLATE_R_WHITE="${TEMPLATE_DIR}/fs_LR.32k.R.white.surf.gii"
TEMPLATE_R_PIAL="${TEMPLATE_DIR}/fs_LR.32k.R.pial.surf.gii"
ATLAS_LABEL="${TEMPLATE_DIR}/Atlas_ROIs.2.nii.gz"
TEMPLATE_DSCALAR="${TEMPLATE_DIR}/91282_Greyordinates.dscalar.nii"
TR_FALLBACK="2.0"

LOGFILE="convert_ds005747_dtseries.log"
: > "$LOGFILE"; exec > >(tee -a "$LOGFILE") 2>&1

echo "=== ds005747 NIfTI → CIFTI dtseries ==="

if ! command -v wb_command >/dev/null 2>&1; then
  echo "wb_command not found in PATH."
  exit 1
fi

for required in \
  "$TEMPLATE_L_MID" "$TEMPLATE_L_WHITE" "$TEMPLATE_L_PIAL" \
  "$TEMPLATE_R_MID" "$TEMPLATE_R_WHITE" "$TEMPLATE_R_PIAL" \
  "$ATLAS_LABEL" "$TEMPLATE_DSCALAR"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required template file: $required"
    exit 1
  fi
done

# Derivatives are named like:
#   sub-011_task-rest_space-MNI305_preproc.nii.gz
mapfile -t FILES < <(find "$INPUT_DIR" -type f -name "sub-*_task-rest_*_preproc.nii.gz" | sort)
TOTAL=${#FILES[@]}
echo "Found $TOTAL input files."
if [[ "$TOTAL" -eq 0 ]]; then
  echo "No files found in $INPUT_DIR matching sub-*_task-rest_*_preproc.nii.gz"
  exit 1
fi

# get_tr() {
#   local nii="$1" tr=""
#   if command -v fslval >/dev/null 2>&1; then
#     tr=$(fslval "$nii" pixdim4 2>/dev/null | awk '{print $1}' || true)
#   fi
#   [[ -z "$tr" ]] && tr="$TR_FALLBACK"
#   printf "%s" "$tr"
# }

SUCCESS=0; SKIPPED=0; FAILED=0; COUNT=0

for nii in "${FILES[@]}"; do
  COUNT=$((COUNT+1))
  rel="${nii#$INPUT_DIR/}"
  bn="$(basename "$nii")"
  stem="${bn%.nii.gz}"
  subject="${stem%%_*}"

  # Keep per-subject outputs to avoid collisions.
  out_dir="$CIFTI_DIR/$subject"
  mkdir -p "$out_dir"

  base="$stem"
  left_gii="$out_dir/${base}.L.func.gii"
  right_gii="$out_dir/${base}.R.func.gii"
  resampled="$out_dir/${base}.resampled.nii.gz"
  dtseries="$out_dir/${base}.dtseries.nii"

  echo "[$COUNT/$TOTAL] $rel"

  [[ -f "$dtseries" ]] && { echo "  skip: dtseries exists"; SKIPPED=$((SKIPPED+1)); echo; continue; }

#   TR=$(get_tr "$nii"); echo "  TR=${TR}s"

  # Surface mapping
  if [[ ! -f "$left_gii" || ! -f "$right_gii" ]]; then
    echo "  map→surfaces"
    wb_command -volume-to-surface-mapping "$nii" "$TEMPLATE_L_MID" "$left_gii" \
      -ribbon-constrained "$TEMPLATE_L_WHITE" "$TEMPLATE_L_PIAL"
    wb_command -volume-to-surface-mapping "$nii" "$TEMPLATE_R_MID" "$right_gii" \
      -ribbon-constrained "$TEMPLATE_R_WHITE" "$TEMPLATE_R_PIAL"
  else
    echo "  surfaces exist"
  fi

  # Volume resample to atlas grid
  if [[ ! -f "$resampled" ]]; then
    echo "  resample→subcortex grid"
    wb_command -volume-resample "$nii" "$ATLAS_LABEL" CUBIC "$resampled"
  else
    echo "  resampled exists"
  fi

  # Build dtseries
  echo "  build dtseries"
  if wb_command -cifti-create-dense-from-template \
        "$TEMPLATE_DSCALAR" "$dtseries" \
        -series 2 0 \
        -volume-all "$resampled" \
        -metric CORTEX_LEFT  "$left_gii" \
        -metric CORTEX_RIGHT "$right_gii"; then
    echo "  ✔ created: ${dtseries#$CIFTI_DIR/}"; SUCCESS=$((SUCCESS+1))
  else
    echo "  ✖ failed"; FAILED=$((FAILED+1))
  fi
  echo
done

echo "Done. Total=$TOTAL Success=$SUCCESS Skipped=$SKIPPED Failed=$FAILED (log: $LOGFILE)"
