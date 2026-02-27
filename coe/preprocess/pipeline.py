#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Note: nibabel imported inside functions using it to allow graceful failure
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # Fallback if tqdm not installed

# Fallback path logic
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from coe.parcellation.parcellate_single import load_cifti_data, process_single_file


def extract_tr(nifti_path: Path) -> float:
    """Extract TR (step) dynamically using nibabel."""
    import nibabel as nib
    try:
        nii = nib.load(str(nifti_path))
        zooms = nii.header.get_zooms()
        if len(zooms) >= 4:
            tr = float(zooms[3])
            return tr if tr > 0 else 1.0
        else:
            print(f"Warning: No time dimension found in {nifti_path.name}")
            return 1.0
    except Exception as e:
        print(f"Error extracting TR from {nifti_path.name}: {e}")
        return 1.0


def map_nifti_to_cifti(
    input_nii: Path, 
    subject_out_dir: Path, 
    template_dir: Path, 
    tr: float
) -> Path:
    """
    Simulates the steps from convert_dtseries.sh:
    1. Maps volume to left & right cortical surfaces.
    2. Resamples volume to subcortical grid.
    3. Combines into dense .dtseries.nii (using the extracted TR).
    """
    subject_out_dir.mkdir(parents=True, exist_ok=True)
    
    stem = input_nii.name.replace(".nii.gz", "").replace(".nii", "")
    
    t_l_mid = template_dir / "fs_LR.32k.L.midthickness.surf.gii"
    t_l_white = template_dir / "fs_LR.32k.L.white.surf.gii"
    t_l_pial = template_dir / "fs_LR.32k.L.pial.surf.gii"
    t_r_mid = template_dir / "fs_LR.32k.R.midthickness.surf.gii"
    t_r_white = template_dir / "fs_LR.32k.R.white.surf.gii"
    t_r_pial = template_dir / "fs_LR.32k.R.pial.surf.gii"
    
    atlas_label = template_dir / "Atlas_ROIs.2.nii.gz"
    template_dscalar = template_dir / "91282_Greyordinates.dscalar.nii"

    # Outputs
    left_gii = subject_out_dir / f"{stem}.L.func.gii"
    right_gii = subject_out_dir / f"{stem}.R.func.gii"
    resampled = subject_out_dir / f"{stem}.resampled.nii.gz"
    dtseries = subject_out_dir / f"{stem}.dtseries.nii"

    if dtseries.exists():
        return dtseries

    DEVNULL = subprocess.DEVNULL

    try:
        if not left_gii.exists():
            subprocess.run([
                "wb_command", "-volume-to-surface-mapping", str(input_nii), str(t_l_mid), str(left_gii),
                "-ribbon-constrained", str(t_l_white), str(t_l_pial)
            ], check=True, stdout=DEVNULL, stderr=DEVNULL)
        
        if not right_gii.exists():
            subprocess.run([
                "wb_command", "-volume-to-surface-mapping", str(input_nii), str(t_r_mid), str(right_gii),
                "-ribbon-constrained", str(t_r_white), str(t_r_pial)
            ], check=True, stdout=DEVNULL, stderr=DEVNULL)

        if not resampled.exists():
            subprocess.run([
                "wb_command", "-volume-resample", str(input_nii), str(atlas_label), "CUBIC", str(resampled)
            ], check=True, stdout=DEVNULL, stderr=DEVNULL)

        # Dense Create (Injecting dynamic TR!)
        subprocess.run([
            "wb_command", "-cifti-create-dense-from-template",
            str(template_dscalar), str(dtseries),
            "-series", str(tr), "0",
            "-volume-all", str(resampled),
            "-metric", "CORTEX_LEFT", str(left_gii),
            "-metric", "CORTEX_RIGHT", str(right_gii)
        ], check=True, stdout=DEVNULL, stderr=DEVNULL)

        return dtseries

    except subprocess.CalledProcessError as e:
        print(f"wb_command failed for {stem}")
        return None


def main():
    parser = argparse.ArgumentParser(description="End-to-End fMRI Preprocessing Pipeline (NIfTI -> CIFTI -> Parcellation)")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing input NIfTI files")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory where final parcellated files will be saved")
    parser.add_argument("--atlas", required=True, type=Path, help="Path to the atlas .dlabel.nii file for parcellation")
    parser.add_argument("--pattern", default="*.nii.gz", type=str, help="Glob pattern to find input files (default: *.nii.gz)")
    parser.add_argument("--template-dir", default=Path(__file__).resolve().parent.parent, type=Path, help="Path to directory containing fs_LR 32k templates.")
    
    args = parser.parse_args()

    if not args.atlas.exists():
        print(f"Error: Atlas file {args.atlas} not found!")
        sys.exit(1)

    print(f"Loading atlas into memory: {args.atlas.name}")
    atlas_data, _ = load_cifti_data(str(args.atlas))
    if atlas_data is None:
        print("Failed to load atlas data.")
        sys.exit(1)
    
    atlas_name = args.atlas.stem.replace(".dlabel", "")

    input_files = list(args.input_dir.rglob(args.pattern))
    if not input_files:
        print(f"No files matching '{args.pattern}' found in {args.input_dir}")
        sys.exit(1)
    
    print(f"Found {len(input_files)} input NIfTI files.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0

    progress = tqdm(input_files, desc="Processing pipeline") if "tqdm" in sys.modules else input_files

    for input_nii in progress:
        subject_id = input_nii.name.split("_")[0]
        
        subject_out_dir = args.output_dir / subject_id
        
        stem = input_nii.name.replace(".nii.gz", "").replace(".nii", "")
        final_out_path = subject_out_dir / f"{stem}_{atlas_name}_parcellated.dtseries.nii"
        
        if final_out_path.exists():
            success_count += 1
            continue
        
        try:
            tr = extract_tr(input_nii)
        except Exception:
            tr = 1.0

        dtseries_path = map_nifti_to_cifti(input_nii, subject_out_dir, args.template_dir, tr)
        if not dtseries_path:
            fail_count += 1
            continue
            
        success = process_single_file(str(dtseries_path), str(final_out_path), atlas_data, tr)
        if success:
            success_count += 1
        else:
            fail_count += 1

    print("\n=== Pipeline Complete ===")
    print(f"Total processed: {len(input_files)}")
    print(f"Successfully parcellated: {success_count}")
    print(f"Failed: {fail_count}")


if __name__ == "__main__":
    main()
