"""
Master Execution Orchestrator for AGRI-FUSE Research Platform.
Runs end-to-end data validation, model benchmarking, neural architecture training,
ablation studies, cross-dataset transfer, and explainability generation.
"""

import sys
import os
import subprocess
import time

# Ensure running from repository root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)

def run_step(step_name, script_path):
    # Resolve script path whether called with repo prefix or bare relative path
    if not os.path.exists(script_path):
        if script_path.startswith("agri-fuse/") and os.path.exists(script_path[len("agri-fuse/"):]):
            script_path = script_path[len("agri-fuse/"):]
        elif os.path.exists(os.path.join("agri-fuse", script_path)):
            script_path = os.path.join("agri-fuse", script_path)

    print(f"\n========================================================")
    print(f"STEP: {step_name}")
    print(f"Script: {script_path}")
    print(f"========================================================")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{os.environ.get('PYTHONPATH', '')}"
    env["OMP_NUM_THREADS"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    res = subprocess.run([sys.executable, script_path], capture_output=False, env=env)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"ERROR: {step_name} failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"SUCCESS: {step_name} completed in {elapsed:.2f} seconds.")

def main():
    print("*******************************************************************")
    print("AGRI-FUSE: Adaptive Gated Multi-Dataset Agricultural Framework")
    print("Master Execution Pipeline Starting on Apple Silicon M2 Environment")
    print("*******************************************************************")
    start_total = time.time()
    
    # Phase 1 & 2: Acquisition, Audit, Inventory
    run_step("Data Acquisition & Status Audit", "src/data/download_datasets.py")
    run_step("Dataset Inspection", "src/data/inspect_datasets.py")
    run_step("Scientific Data Validation", "src/data/validate_datasets.py")
    run_step("Generate Inventory & Provenance", "src/data/generate_inventory.py")
    run_step("Generate Feature Dictionary", "src/data/generate_feature_dictionary.py")
    run_step("Dataset Compatibility Matrix", "src/data/dataset_compatibility.py")
    
    # Phase 3: Data Splitting (Leakage Prevention)
    run_step("Reproducible Data Splitting", "src/preprocessing/data_splitters.py")
    
    # Phase 4 & 5 & 6: Experimental Pipelines
    run_step("Pipeline A: Crop Recommendation Experiments", "experiments/crop_recommendation/run_crop_rec_experiments.py")
    run_step("Pipeline B: Historical Macro Yield Experiments", "experiments/yield_prediction/run_yield_experiments.py")
    run_step("Pipeline C: Field-Level Wheat Yield Experiments", "experiments/wheat_analysis/run_wheat_experiments.py")
    run_step("Pipeline D: Smart Farming IoT Experiments", "experiments/smart_farming_iot/run_iot_experiments.py")
    run_step("Pipeline E: Satellite-Derived Vegetation Index Experiments", "experiments/satellite_indices/run_satellite_experiments.py")
    
    # Phase 7: Ablations & Cross-Dataset Generalization
    run_step("Ablation Studies & Cross-Dataset Transfer", "experiments/run_ablations_and_generalization.py")
    
    # Phase 8: Explainability & XAI
    run_step("Explainable AI & Gate-Weight Visualizations", "src/explainability/run_explainability.py")
    
    # Phase 9: AGRI-FUSE Hybrid Ensemble (AF-HE) Benchmarks
    run_step("AF-HE Hybrid Ensemble (Stacking, Blending, Ablations, Calibration)", "experiments/run_af_he_experiments.py")
    
    total_time = time.time() - start_total
    print("\n*******************************************************************")
    print(f"ALL EXPERIMENTS COMPLETED SUCCESSFULLY IN {total_time/60:.2f} MINUTES!")
    print("*******************************************************************")

if __name__ == "__main__":
    main()
