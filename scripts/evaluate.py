import argparse
import sys
import torch
import warnings
import logging
from pathlib import Path

# Setup logging filters to avoid filling terminal with library warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger("guitar_tutor_pipeline.src.app.model_registry").setLevel(logging.ERROR)
logging.getLogger("guitar_tutor_pipeline.src.app.alignment").setLevel(logging.ERROR)
logging.getLogger("guitar_tutor_pipeline.src.app.pipeline").setLevel(logging.ERROR)
logging.getLogger("guitar_tutor_pipeline.src.TabCNN_Architecture.inference").setLevel(logging.ERROR)

from eval_utils import get_test_files, run_evaluation, save_results

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    try:
        if torch.backends.mps.is_available():
            return 'mps'
    except AttributeError:
        pass
    return 'cpu'

def main():
    parser = argparse.ArgumentParser(description="Evaluate transcription models on test datasets.")
    parser.add_argument("--models", type=str, choices=["CRNN", "TabCNN", "both"], default="both",
                        help="Models to evaluate (default: both)")
    parser.add_argument("--dataset", type=str, choices=["synthetic", "real", "both"], default="both",
                        help="Dataset type to evaluate (default: both)")
    parser.add_argument("--out-name", type=str, default="eval",
                        help="Prefix for the output files (default: eval)")
                        
    args = parser.add_argument_group() # Ignore
    args = parser.parse_args()
    
    models_to_eval = ["CRNN", "TabCNN"] if args.models == "both" else [args.models]
    
    print(f"\n=============================================")
    print(f"🎸 Guitar Tutor AI - Evaluation Script")
    print(f"=============================================")
    print(f" Models selected : {', '.join(models_to_eval)}")
    print(f" Dataset selected: {args.dataset}")
    print(f"=============================================\n")
    
    test_cases = get_test_files(args.dataset)
    print(f"Found {len(test_cases)} test files matching criteria.")
    
    if not test_cases:
        print("No test files found. Exiting.")
        sys.exit(0)
        
    device = get_device()
    print(f"Using device for inference: {device}\n")
    
    print("Running evaluation (this may take a while)...")
    results = run_evaluation(test_cases, models_to_eval, device=device)
    
    out_dir = Path(__file__).resolve().parent / "out"
    j_path, c_path = save_results(results, out_dir, args.out_name)
    
    # Print a summary table to the terminal
    print("\n--- Summary of Results ---")
    print(f"{'File':<40} | {'Type':<10} | {'Model':<6} | {'F1 (%)':>6} | {'P (%)':>6} | {'R (%)':>6} |")
    print("-" * 88)
    for r in results:
        # Avoid breaking the table if filename is too long
        file_disp = (r['file'][:37] + "...") if len(r['file']) > 40 else r['file']
        print(f"{file_disp:<40} | {r['dataset_type']:<10} | {r['model']:<6} | {r['f1']:6.1f} | {r['precision']:6.1f} | {r['recall']:6.1f} |")
        
    print(f"\n✅ Results saved successfully to:")
    print(f" - JSON: {j_path}")
    print(f" - CSV : {c_path}\n")

if __name__ == "__main__":
    main()
