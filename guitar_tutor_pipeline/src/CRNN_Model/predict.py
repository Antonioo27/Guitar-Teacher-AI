import argparse
from pathlib import Path
import torch
import pretty_midi
import numpy as np
import mir_eval

from .config import config
from .model import Regress_onset_offset_frame_velocity_CRNN, load_finetuned_checkpoint, load_maestro_checkpoint
from .inference import transcribe_full

def main():
    parser = argparse.ArgumentParser(description="Trascrive un file .wav e lo confronta con un file .mid di riferimento usando il modello CRNN.")
    parser.add_argument("wav_path", type=str, help="Percorso del file audio .wav da trascrivere")
    parser.add_argument("midi_path", type=str, nargs="?", help="[Opzionale] Percorso del file .mid di riferimento per calcolare le metriche")
    parser.add_argument("--checkpoint", type=str, default=str(config.MAESTRO_CHECKPOINT), help=f"Percorso del modello pre-addestrato (default: {config.MAESTRO_CHECKPOINT})")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu o cuda)")
    
    args = parser.parse_args()

    wav_path = Path(args.wav_path)
    if not wav_path.exists():
        print(f"Errore: Il file {wav_path} non esiste.")
        return

    # Inizializza il modello
    print(f"Inizializzazione modello su {args.device}...")
    model = Regress_onset_offset_frame_velocity_CRNN(
        frames_per_second=config.FRAMES_PER_SECOND,
        classes_num=config.CLASSES_NUM
    ).to(args.device)

    # Carica i pesi
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        try:
            # Prova a caricare come checkpoint fine-tuned
            model = load_finetuned_checkpoint(model, checkpoint_path, device=args.device)
            print("Checkpoint fine-tuned caricato con successo.")
        except Exception:
            # Fallback a MAESTRO checkpoint
            model = load_maestro_checkpoint(model, checkpoint_path, device=args.device)
            print("Checkpoint MAESTRO caricato con successo.")
    else:
        print(f"Errore: Checkpoint non trovato in {checkpoint_path}")
        return

    # Esegui l'inferenza
    print(f"\nTrascrizione del file {wav_path} in corso...")
    predicted_notes = transcribe_full(model, wav_path, args.device)
    print(f"Trascrizione completata: {len(predicted_notes)} note rilevate.")

    # Se è fornito un file MIDI, calcola le metriche
    if args.midi_path:
        midi_path = Path(args.midi_path)
        if not midi_path.exists():
            print(f"Attenzione: Il file MIDI {midi_path} non esiste. Impossibile calcolare le metriche.")
            return

        print(f"\nConfronto con {midi_path}...")
        ref_pm = pretty_midi.PrettyMIDI(str(midi_path))
        ref_notes = [(n.start, n.end, n.pitch) for inst in ref_pm.instruments
                     for n in inst.notes if not inst.is_drum]
        
        if len(ref_notes) == 0:
            print("Il file MIDI di riferimento non contiene note (o solo batteria).")
            return

        ref_int = np.array([[s, e] for s, e, _ in ref_notes])
        ref_pit = np.array([p for _, _, p in ref_notes])
        est_int = np.array([[s, e] for s, e, _, _ in predicted_notes]) if predicted_notes else np.empty((0, 2))
        est_pit = np.array([p for _, _, p, _ in predicted_notes]) if predicted_notes else np.empty(0)

        if len(predicted_notes) > 0:
            P, R, F1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_int, ref_pit, est_int, est_pit,
                onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
            
            print("\n" + "="*40)
            print("RISULTATI VALUTAZIONE (MIR_EVAL)")
            print("="*40)
            print(f"Note Reference: {len(ref_notes)}")
            print(f"Note Predette : {len(predicted_notes)}")
            print(f"Precision (P) : {P*100:.2f}%")
            print(f"Recall (R)    : {R*100:.2f}%")
            print(f"F1 Score      : {F1*100:.2f}%")
            print("="*40)
        else:
            print("Il modello non ha rilevato alcuna nota, F1 Score = 0.00%")
            
    else:
        # Se non c'è il MIDI, mostra solo le prime 10 note predette
        print("\nPrime 10 note predette (start, end, pitch, velocity):")
        for n in predicted_notes[:10]:
            print(f"  {n[0]:.2f}s - {n[1]:.2f}s | Pitch: {n[2]} | Vel: {n[3]}")

if __name__ == "__main__":
    main()
