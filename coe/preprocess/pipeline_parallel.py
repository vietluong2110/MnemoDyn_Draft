#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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

    try:
        if not left_gii.exists():
            subprocess.run([
                "wb_command", "-volume-to-surface-mapping", str(input_nii), str(t_l_mid), str(left_gii),
                "-ribbon-constrained", str(t_l_white), str(t_l_pial)
            ], check=True, capture_output=True, text=True)
        
        if not right_gii.exists():
            subprocess.run([
                "wb_command", "-volume-to-surface-mapping", str(input_nii), str(t_r_mid), str(right_gii),
                "-ribbon-constrained", str(t_r_white), str(t_r_pial)
            ], check=True, capture_output=True, text=True)

        if not resampled.exists():
            subprocess.run([
                "wb_command", "-volume-resample", str(input_nii), str(atlas_label), "CUBIC", str(resampled)
            ], check=True, capture_output=True, text=True)

        # Dense Create (Injecting dynamic TR!)
        subprocess.run([
            "wb_command", "-cifti-create-dense-from-template",
            str(template_dscalar), str(dtseries),
            "-series", str(tr), "0",
            "-volume-all", str(resampled),
            "-metric", "CORTEX_LEFT", str(left_gii),
            "-metric", "CORTEX_RIGHT", str(right_gii)
        ], check=True, capture_output=True, text=True)

        return dtseries

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] wb_command failed for {stem}")
        print(f"Command run: {' '.join(e.cmd)}")
        print(f"Error output:\n{e.stderr}")
        return None

def process_subject(input_nii: Path, output_dir: Path, template_dir: Path, atlas_data, atlas_name: str) -> bool:
    """Wrapper function to process a single subject end-to-end, meant for multiprocessing."""
    subject_id = input_nii.name.split("_")[0]
    subject_out_dir = output_dir / subject_id
    
    stem = input_nii.name.replace(".nii.gz", "").replace(".nii", "")
    final_out_path = subject_out_dir / f"{stem}_{atlas_name}_parcellated.dtseries.nii"
    
    if final_out_path.exists():
        return True # already processed
        
    try:
        tr = extract_tr(input_nii)
    except Exception:
        tr = 1.0

    dtseries_path = map_nifti_to_cifti(input_nii, subject_out_dir, template_dir, tr)
    if not dtseries_path:
        return False
        
    success = process_single_file(str(dtseries_path), str(final_out_path), atlas_data, tr)
    return success


def main():
    parser = argparse.ArgumentParser(description="Parallelized End-to-End fMRI Preprocessing Pipeline")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing input NIfTI files")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory where final parcellated files will be saved")
    parser.add_argument("--atlas", required=True, type=Path, help="Path to the atlas .dlabel.nii file for parcellation")
    parser.add_argument("--pattern", default="*.nii.gz", type=str, help="Glob pattern to find input files (default: *.nii.gz)")
    parser.add_argument("--template-dir", default=Path(__file__).resolve().parent.parent, type=Path, help="Path to directory containing fs_LR 32k templates.")
    parser.add_argument("--jobs", type=int, default=None, help="Number of concurrent processes (default: use all available CPU cores). Note: High RAM required.")
    
    args = parser.parse_args()

    if not args.atlas.exists():
        print(f"Error: Atlas file {args.atlas} not found!")
        sys.exit(1)

    print(f"Loading atlas into memory: {args.atlas.name}...")
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
    
    # Determine jobs
    max_workers = args.jobs if args.jobs is not None else os.cpu_count()
    print(f"Starting multiprocessing pool with {max_workers} concurrent jobs...")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0

    # Start parallel execution
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Map futures to subject inputs
        future_to_nii = {
            executor.submit(process_subject, nii, args.output_dir, args.template_dir, atlas_data, atlas_name): nii 
            for nii in input_files
        }
        
        # Use simple progress bar if tqdm not installed, else use tqdm
        if "tqdm" in sys.modules:
            completed_futures = tqdm(as_completed(future_to_nii), total=len(input_files), desc="Parallel Processing")
        else:
            completed_futures = as_completed(future_to_nii)
            
        for future in completed_futures:
            nii = future_to_nii[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as exc:
                print(f"[ERROR] Subject {nii.name} generated an exception: {exc}")
                fail_count += 1

    print("\n=== Parallel Pipeline Complete ===")
    print(f"Total processed: {len(input_files)}")
    print(f"Successfully parcellated: {success_count}")
    print(f"Failed: {fail_count}")


if __name__ == "__main__":
    main()
