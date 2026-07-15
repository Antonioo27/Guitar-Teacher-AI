import os
import torch
import numpy as np
import pretty_midi
import mir_eval
from guitar_tutor_pipeline.src.CRNN_Model.config import config
from guitar_tutor_pipeline.src.CRNN_Model.model import Regress_onset_offset_frame_velocity_CRNN, load_finetuned_checkpoint
from guitar_tutor_pipeline.src.CRNN_Model.inference import get_activations, decode_activations, filter_short_notes
from guitar_tutor_pipeline.src.CRNN_Model.split import load_pairs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Regress_onset_offset_frame_velocity_CRNN(config.FRAMES_PER_SECOND, config.CLASSES_NUM).to(device)
    model = load_finetuned_checkpoint(model, config.BEST_MODEL_PATH, device=device)

    # Prendi alcuni file dal test set
    test_pairs = load_pairs("test")[:5]

    print("\nVerifica Ipotesi: Errore di timing umano vs Modello")
    print("-" * 60)
    print(f"{'File':<20} | {'F1 (Strict 50ms)':<16} | {'F1 (Loose 500ms)':<16}")
    print("-" * 60)

    for wav, mid in test_pairs:
        segs = get_activations(model, wav, device)
        est = decode_activations(segs, onset_thresh=0.05)
        est = filter_short_notes(est)
        
        ref_pm = pretty_midi.PrettyMIDI(str(mid))
        ref = [(n.start, n.end, n.pitch) for inst in ref_pm.instruments for n in inst.notes if not inst.is_drum]
        
        if not ref or not est:
            continue
            
        ref_int = np.array([[s, e] for s, e, _ in ref]); ref_pit = np.array([p for _, _, p in ref])
        est_int = np.array([[s, e] for s, e, _, _ in est]); est_pit = np.array([p for _, _, p, _ in est])
        
        _, _, f1_strict, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_int, ref_pit, est_int, est_pit, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
            
        _, _, f1_loose, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_int, ref_pit, est_int, est_pit, onset_tolerance=0.5, pitch_tolerance=50.0, offset_ratio=None)
            
        print(f"{wav.name:<20} | {f1_strict*100:>15.1f}% | {f1_loose*100:>15.1f}%")

if __name__ == '__main__':
    main()
