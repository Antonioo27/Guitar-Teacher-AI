# evaluate.py
import os
import time
import numpy as np
import torch
import pretty_midi
import mir_eval

from guitar_tutor_pipeline.src.CRNN_Model.config import config
from guitar_tutor_pipeline.src.CRNN_Model.model import (
    Regress_onset_offset_frame_velocity_CRNN,
    load_finetuned_checkpoint,
)
from guitar_tutor_pipeline.src.CRNN_Model.inference import (
    get_activations,
    decode_activations,
    filter_short_notes
)
from guitar_tutor_pipeline.src.CRNN_Model.split import load_pairs

# su CPU usa tutti i core (a volte di default ne usa 1 solo)
torch.set_num_threads(os.cpu_count() or 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def precompute(model, pairs, device, label=""):
    """Forward UNA volta per file. Stampa avanzamento, durata per file ed ETA."""
    cache = []
    n_files = len(pairs)
    t_start = time.time()
    for i, (wav, mid) in enumerate(pairs, 1):
        t0 = time.time()
        segs = get_activations(model, wav, device)
        dt = time.time() - t0
        elapsed = time.time() - t_start
        eta = elapsed / i * (n_files - i)          # stima sui file già fatti
        print(f"  [{label}] {i:>2}/{n_files}  {wav.name:<18} "
              f"{len(segs):>2} seg  {dt:5.1f}s | tot {elapsed:6.1f}s  ETA ~{eta:5.0f}s",
              flush=True)

        ref_pm = pretty_midi.PrettyMIDI(str(mid))
        ref = [(nt.start, nt.end, nt.pitch) for inst in ref_pm.instruments
               for nt in inst.notes if not inst.is_drum]
        if not ref:
            print(f"      (nessuna nota di riferimento in {mid.name}, salto)", flush=True)
            continue
        ref_int = np.array([[s, e] for s, e, _ in ref])
        ref_pit = np.array([p for _, _, p in ref])
        cache.append((wav.name, segs, ref_int, ref_pit))

    print(f"  [{label}] forward completati in {time.time()-t_start:.1f}s "
          f"({len(cache)}/{n_files} file validi)\n", flush=True)
    return cache


def prf_from_cache(entry, onset_thresh, frame_thresh=0.3):
    name, segs, ref_int, ref_pit = entry
    est = decode_activations(segs, onset_thresh, frame_thresh)
    if not est:
        return None
    est_int = np.array([[s, e] for s, e, _, _ in est])
    est_pit = np.array([p for _, _, p, _ in est])
    P, R, F1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_pit, est_int, est_pit,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    return P, R, F1

# in evaluate.py o in un nuovo confronto_reale.py
def prf_real(model, wav_path, ref_midi, device, onset_thresh=0.05):
    from guitar_tutor_pipeline.src.CRNN_Model.inference import get_activations, decode_activations
    segs = get_activations(model, wav_path, device)
    est = decode_activations(segs, onset_thresh=onset_thresh)
    est = filter_short_notes(est)                      # togli i fantasmi

    ref_pm = pretty_midi.PrettyMIDI(str(ref_midi))
    ref = [(n.start, n.end, n.pitch) for inst in ref_pm.instruments
           for n in inst.notes if not inst.is_drum]
    if not ref or not est:
        return None
    ref_int = np.array([[s, e] for s, e, _ in ref]); ref_pit = np.array([p for _, _, p in ref])
    est_int = np.array([[s, e] for s, e, _, _ in est]); est_pit = np.array([p for _, _, p, _ in est])
    P, R, F1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_pit, est_int, est_pit,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    return P, R, F1


if __name__ == "__main__":
    print(f"Device: {device}  |  thread CPU: {torch.get_num_threads()}\n", flush=True)

    print("Caricamento modello fine-tuned...", flush=True)
    model = Regress_onset_offset_frame_velocity_CRNN(
        config.FRAMES_PER_SECOND, config.CLASSES_NUM).to(device)
    model = load_finetuned_checkpoint(model, config.BEST_MODEL_PATH, device=device)

    dev_pairs, test_pairs = load_pairs("dev"), load_pairs("test")
    print(f"Caricati: dev={len(dev_pairs)}  test={len(test_pairs)}\n", flush=True)

    # ---------- FASE 1: forward sui DEV ----------
    print("=== FASE 1/3: forward DEV ===", flush=True)
    dev_cache = precompute(model, dev_pairs, device, label="DEV")

    # ---------- FASE 2: sweep soglia (solo decode, veloce) ----------
    print("=== FASE 2/3: scelta soglia su DEV ===", flush=True)
    grid = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]
    dev_scores = {}
    for th in grid:
        f1s = [r[2] for e in dev_cache if (r := prf_from_cache(e, th))]
        dev_scores[th] = float(np.mean(f1s)) if f1s else 0.0
        print(f"  thresh={th:.2f}  F1_dev={dev_scores[th]*100:5.1f}", flush=True)
    best_th = max(dev_scores, key=dev_scores.get)
    print(f"\n>>> Soglia scelta: {best_th}  (F1_dev={dev_scores[best_th]*100:.1f})\n", flush=True)

    # ---------- FASE 3: forward sui TEST + numero finale ----------
    print("=== FASE 3/3: forward TEST + valutazione finale ===", flush=True)
    test_cache = precompute(model, test_pairs, device, label="TEST")

    rows = [(e[0], prf_from_cache(e, best_th)) for e in test_cache]
    rows = [(name, r) for name, r in rows if r is not None]
    P = np.array([r[1][0] for r in rows])
    R = np.array([r[1][1] for r in rows])
    F = np.array([r[1][2] for r in rows])

    print(f"\n{'file':<20} {'P':>6} {'R':>6} {'F1':>6}")
    for name, (p, r, f) in rows:
        print(f"{name:<20} {p*100:6.1f} {r*100:6.1f} {f*100:6.1f}")
    print("=" * 42)
    print(f"TEST ({len(F)} file)  "
          f"P={P.mean()*100:.1f}±{P.std()*100:.1f}  "
          f"R={R.mean()*100:.1f}±{R.std()*100:.1f}  "
          f"F1={F.mean()*100:.1f}±{F.std()*100:.1f}")