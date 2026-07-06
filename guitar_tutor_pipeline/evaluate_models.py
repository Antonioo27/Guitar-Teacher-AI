import sys
import logging
import warnings
from pathlib import Path
import numpy as np
import torch

# Ignora i RuntimeWarning di pretty_midi
warnings.filterwarnings("ignore")

# Disabilita i log INFO/DEBUG e i WARNING per avere una tabella pulita a terminale
logging.basicConfig(level=logging.ERROR)
logging.getLogger("guitar_tutor_pipeline.src.app.model_registry").setLevel(logging.ERROR)
logging.getLogger("guitar_tutor_pipeline.src.app.alignment").setLevel(logging.ERROR)
logging.getLogger("guitar_tutor_pipeline.src.app.pipeline").setLevel(logging.ERROR)
logging.getLogger("guitar_tutor_pipeline.src.TabCNN_Architecture.inference").setLevel(logging.ERROR)

from guitar_tutor_pipeline.src.app.model_registry import load_model, transcribe
from guitar_tutor_pipeline.src.app.dataset import parse_midi, build_note_sequence
from guitar_tutor_pipeline.src.app.alignment import run_alignment

def main():
    # File di output
    output_file = Path("evaluation_results.txt")
    
    # Helper per stampare a schermo e scrivere su file contemporaneamente
    def print_and_save(text="", end="\n"):
        print(text, end=end)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(text + end)
            
    # Svuota o crea il file all'inizio
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_and_save(f"Device per l'inferenza: {device}\n")
    
    print_and_save("Caricamento modelli in corso...")
    try:
        model_crnn = load_model("CRNN", device)
        model_tabcnn = load_model("TabCNN", device)
    except Exception as e:
        print_and_save(f"Errore critico durante il caricamento dei modelli: {e}")
        sys.exit(1)
        
    models = {
        "CRNN": model_crnn,
        "TabCNN": model_tabcnn
    }
    
    # Risolvi la path della cartella dei test rispetto alla root del progetto
    base_dir = Path(__file__).resolve().parent.parent / "SetTestHomeMade"
    true_dir = base_dir / "TrueRecordings"
    
    if not base_dir.exists():
        print_and_save(f"Directory non trovata: {base_dir}")
        sys.exit(1)
        
    midis = sorted(base_dir.glob("*.mid"))
    if not midis:
        print_and_save(f"Nessun file .mid trovato in {base_dir}")
        sys.exit(0)
        
    print_and_save(f"Trovati {len(midis)} file .mid di riferimento in SetTestHomeMade.\n")
    
    results = [] # Lista per salvare le metriche: (Type, Model, P, R, F1)
    
    print_and_save(f"{'File':<40} | {'Type':<10} | {'Model':<6} | {'P (%)':>6} | {'R (%)':>6} | {'F1 (%)':>6} | {'Offset (s)':>10} |")
    print_and_save("-" * 108)
    
    for midi_path in midis:
        stem = midi_path.stem
        # Identifica il prefisso numerico per raggruppare i file reali (es. "00_", "03_")
        prefix = stem.split("_")[0] + "_" 
        
        try:
            ref_annotations = parse_midi(str(midi_path))
            reference_notes = build_note_sequence(ref_annotations)
        except Exception as e:
            print_and_save(f"Errore lettura {midi_path.name}: {e}")
            continue
            
        # =======================================================
        # 1. Valutazione su Audio Sintetico (stessa dir)
        # =======================================================
        synth_wav = base_dir / (stem + ".wav")
        if synth_wav.exists():
            for m_name, m_obj in models.items():
                try:
                    # Trascrizione
                    pred_notes = transcribe(str(synth_wav), m_obj, m_name, device)
                    # Allineamento e Metriche (calcolate da mir_eval tramite alignment.py)
                    error_log = run_alignment(pred_notes, reference_notes)
                    summary = error_log["summary"]
                    
                    P = summary.get("precision_percent", 0.0)
                    R = summary.get("recall_percent", 0.0)
                    F1 = summary.get("accuracy_percent", 0.0)
                    offset = summary.get("estimated_global_offset_sec", 0.0)
                    
                    print_and_save(f"{synth_wav.stem[:40]:<40} | {'Synthetic':<10} | {m_name:<6} | {P:6.1f} | {R:6.1f} | {F1:6.1f} | {offset:10.3f} |")
                    results.append(("Synthetic", m_name, P, R, F1))
                except Exception as e:
                    print_and_save(f"Errore evaluation synthetic {synth_wav.name} con {m_name}: {e}")
        
        # =======================================================
        # 2. Valutazione su Audio Reale (TrueRecordings)
        # =======================================================
        true_wavs = sorted(true_dir.glob(f"{prefix}*.wav"))
        for t_wav in true_wavs:
            for m_name, m_obj in models.items():
                try:
                    # Trascrizione
                    pred_notes = transcribe(str(t_wav), m_obj, m_name, device)
                    # Allineamento e Metriche
                    error_log = run_alignment(pred_notes, reference_notes)
                    summary = error_log["summary"]
                    
                    P = summary.get("precision_percent", 0.0)
                    R = summary.get("recall_percent", 0.0)
                    F1 = summary.get("accuracy_percent", 0.0)
                    offset = summary.get("estimated_global_offset_sec", 0.0)
                    
                    print_and_save(f"{t_wav.stem[:40]:<40} | {'Real':<10} | {m_name:<6} | {P:6.1f} | {R:6.1f} | {F1:6.1f} | {offset:10.3f} |")
                    results.append(("Real", m_name, P, R, F1))
                except Exception as e:
                    print_and_save(f"Errore evaluation real {t_wav.name} con {m_name}: {e}")
                    
    # =======================================================
    # Statistiche Aggregate
    # =======================================================
    print_and_save("-" * 108)
    print_and_save("\nMEDIA DELLE METRICHE PER CATEGORIA E MODELLO:")
    print_and_save(f"{'Type':<15} | {'Model':<10} | {'N. File':<7} | {'P (%)':>6} | {'R (%)':>6} | {'F1 (%)':>6} |")
    print_and_save("-" * 70)
    
    for t_type in ["Synthetic", "Real"]:
        for m_name in ["CRNN", "TabCNN"]:
            subset = [r for r in results if r[0] == t_type and r[1] == m_name]
            if not subset:
                continue
            avg_P = np.mean([r[2] for r in subset])
            avg_R = np.mean([r[3] for r in subset])
            avg_F1 = np.mean([r[4] for r in subset])
            print_and_save(f"{t_type:<15} | {m_name:<10} | {len(subset):<7} | {avg_P:6.1f} | {avg_R:6.1f} | {avg_F1:6.1f} |")
            
    print_and_save("-" * 70)
    print(f"\nReport salvato con successo in: {output_file.absolute()}")

if __name__ == '__main__':
    main()
