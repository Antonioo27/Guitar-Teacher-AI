import sys
import json
import csv
from pathlib import Path

# Add project root to path so we can import guitar_tutor_pipeline
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from guitar_tutor_pipeline.src.app.model_registry import load_model, transcribe
from guitar_tutor_pipeline.src.app.dataset import parse_midi, build_note_sequence
from guitar_tutor_pipeline.src.app.alignment import run_alignment

def get_test_files(dataset_type="both"):
    """
    Finds and groups test files into synthetic and real.
    
    Args:
        dataset_type (str): 'synthetic', 'real', or 'both'.
        
    Returns:
        list of dicts containing the file type and paths.
    """
    test_dir = project_root / "SetTestHomeMade"
    true_dir = test_dir / "TrueRecordings"
    
    midis = sorted(test_dir.glob("*.mid"))
    test_cases = []
    
    for midi_path in midis:
        stem = midi_path.stem
        # e.g., '00_CMaj_120' -> prefix '00_'
        prefix = stem.split("_")[0] + "_"
        
        if dataset_type in ["synthetic", "both"]:
            synth_wav = test_dir / f"{stem}.wav"
            if synth_wav.exists():
                test_cases.append({
                    "type": "synthetic",
                    "wav_path": synth_wav,
                    "mid_path": midi_path
                })
        
        if dataset_type in ["real", "both"]:
            if true_dir.exists():
                true_wavs = sorted(true_dir.glob(f"{prefix}*.wav"))
                for t_wav in true_wavs:
                    test_cases.append({
                        "type": "real",
                        "wav_path": t_wav,
                        "mid_path": midi_path
                    })
    return test_cases

def run_evaluation(test_cases, models_to_eval, device="cpu"):
    """
    Runs the transcription models and computes alignment metrics.
    
    Args:
        test_cases (list): List of test cases from get_test_files().
        models_to_eval (list): List of model names to evaluate (e.g. ['CRNN', 'TabCNN']).
        device (str): PyTorch device string ('cpu' or 'cuda').
        
    Returns:
        list of dicts containing the metrics for each file and model.
    """
    models = {}
    for m in models_to_eval:
        models[m] = load_model(m, device)
        
    results = []
    
    for i, case in enumerate(test_cases):
        wav_path = case["wav_path"]
        mid_path = case["mid_path"]
        dataset_type = case["type"]
        
        try:
            ref_annotations = parse_midi(str(mid_path))
            reference_notes = build_note_sequence(ref_annotations)
        except Exception as e:
            print(f"Error reading MIDI {mid_path.name}: {e}")
            continue
            
        for m_name in models_to_eval:
            try:
                m_obj = models[m_name]
                pred_notes = transcribe(str(wav_path), m_obj, m_name, device)
                
                error_log = run_alignment(pred_notes, reference_notes)
                summary = error_log["summary"]
                
                results.append({
                    "file": wav_path.name,
                    "dataset_type": dataset_type,
                    "model": m_name,
                    "precision": summary.get("precision_percent", 0.0),
                    "recall": summary.get("recall_percent", 0.0),
                    "f1": summary.get("accuracy_percent", 0.0),
                    "offset": summary.get("estimated_global_offset_sec", 0.0)
                })
            except Exception as e:
                print(f"Error evaluating {wav_path.name} with {m_name}: {e}")
                
    return results

def save_results(results, output_dir, prefix="evaluation"):
    """
    Saves a list of dictionaries to a JSON file and a CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / f"{prefix}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    csv_path = output_dir / f"{prefix}_results.csv"
    if results:
        keys = results[0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
            
    return json_path, csv_path
