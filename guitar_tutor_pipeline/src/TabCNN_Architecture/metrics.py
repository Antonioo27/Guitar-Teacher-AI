"""
metrics.py — Metriche di accuratezza della predizione vs spartito di riferimento.

Definizione di "accuratezza" adottata (in linea con TabCNN / FretNet / SynthTab,
ma adattata a un riferimento MIDI che NON contiene l'informazione di corda):

  • Note-level (mir_eval.transcription)
      Una nota predetta "matcha" una di riferimento se ha lo STESSO pitch e
      l'onset entro una tolleranza (default 50 ms). Da qui Precision / Recall /
      F1. Due varianti:
        - onset-only            → conta solo l'attacco
        - onset + offset        → conta anche la fine (durata simile)
      Più average overlap ratio (quanto si sovrappongono nel tempo le note matchate).

  • Frame-level pitch (multipitch, stile TabCNN)
      Si "rasterizza" il tempo in frame (default 100 fps): per ogni frame si
      confronta l'insieme dei pitch attivi. Da qui Precision / Recall / F1 e
      l'accuratezza per-frame = TP / (TP + FP + FN).

  • Errori di tempo e durata (sulle note matchate)
      Sfasamento di onset (medio/mediano/dev.std/max assoluto), % note entro
      tolleranza, errore medio di durata.

NOTA: le metriche di tablatura (corda+tasto) e il TDR richiedono un ground
truth con annotazione delle corde (es. GuitarSet .jams). Con un MIDI non sono
calcolabili e qui non vengono riportate.
"""

import json
import logging
import statistics
from pathlib import Path

import numpy as np
import mir_eval

logger = logging.getLogger(__name__)


def _pitch(n: dict) -> int:
    return n.get("pitch", n.get("midi_pitch"))


def _midi_to_hz(m: int) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def _intervals_and_pitches(notes: list[dict]):
    """Converte le note in (intervalli Nx2 [onset,offset], pitch in Hz) per mir_eval."""
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    intervals = np.array(
        [[n["time"], n["time"] + max(n.get("duration", 0.05), 1e-3)] for n in notes]
    )
    pitches = np.array([_midi_to_hz(_pitch(n)) for n in notes])
    return intervals, pitches


# =========================================================================
# Note-level (mir_eval)
# =========================================================================
def note_level_metrics(
    predicted: list[dict],
    reference: list[dict],
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
) -> dict:
    ref_int, ref_p = _intervals_and_pitches(reference)
    est_int, est_p = _intervals_and_pitches(predicted)

    if len(ref_int) == 0 or len(est_int) == 0:
        return {"error": "sequenza vuota: metriche note-level non calcolabili"}

    # Onset-only (offset_ratio=None ignora la fine della nota)
    p_on, r_on, f_on, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_p, est_int, est_p,
        onset_tolerance=onset_tolerance, offset_ratio=None,
    )
    # Onset + offset
    p_off, r_off, f_off, aor = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_p, est_int, est_p,
        onset_tolerance=onset_tolerance, offset_ratio=offset_ratio,
    )

    # Conteggi (basati sul match onset+pitch)
    matches = mir_eval.transcription.match_notes(
        ref_int, ref_p, est_int, est_p,
        onset_tolerance=onset_tolerance, offset_ratio=None,
    )
    tp = len(matches)
    fp = len(predicted) - tp
    fn = len(reference) - tp

    return {
        "onset": {"precision": round(p_on, 4), "recall": round(r_on, 4), "f1": round(f_on, 4)},
        "onset_offset": {"precision": round(p_off, 4), "recall": round(r_off, 4), "f1": round(f_off, 4)},
        "average_overlap_ratio": round(float(aor), 4),
        "counts": {"true_positive": tp, "false_positive": fp, "false_negative": fn},
        "onset_tolerance_s": onset_tolerance,
    }


# =========================================================================
# Frame-level pitch (multipitch, stile TabCNN)
# =========================================================================
def frame_level_metrics(
    predicted: list[dict],
    reference: list[dict],
    fps: int = 100,
) -> dict:
    if not predicted and not reference:
        return {"error": "entrambe le sequenze vuote"}

    def _end(notes):
        return max((n["time"] + max(n.get("duration", 0.05), 1e-3) for n in notes), default=0.0)

    end_time = max(_end(predicted), _end(reference))
    n_frames = int(round(end_time * fps)) + 1

    all_p = [_pitch(n) for n in predicted] + [_pitch(n) for n in reference]
    lo, hi = min(all_p), max(all_p)
    n_pitch = hi - lo + 1

    def _roll(notes):
        grid = np.zeros((n_pitch, n_frames), dtype=bool)
        for n in notes:
            f0 = int(round(n["time"] * fps))
            f1 = int(round((n["time"] + max(n.get("duration", 0.05), 1e-3)) * fps))
            grid[_pitch(n) - lo, f0:max(f1, f0 + 1)] = True
        return grid

    ref_grid = _roll(reference)
    est_grid = _roll(predicted)

    tp = int(np.sum(ref_grid & est_grid))
    fp = int(np.sum(est_grid & ~ref_grid))
    fn = int(np.sum(ref_grid & ~est_grid))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "fps": fps,
        "frames_evaluated": n_frames,
    }


# =========================================================================
# Errori di tempo e durata (sulle note matchate per onset+pitch)
# =========================================================================
def timing_metrics(
    predicted: list[dict],
    reference: list[dict],
    onset_tolerance: float = 0.05,
) -> dict:
    ref_int, ref_p = _intervals_and_pitches(reference)
    est_int, est_p = _intervals_and_pitches(predicted)
    if len(ref_int) == 0 or len(est_int) == 0:
        return {"error": "sequenza vuota"}

    matches = mir_eval.transcription.match_notes(
        ref_int, ref_p, est_int, est_p,
        onset_tolerance=onset_tolerance, offset_ratio=None,
    )
    if not matches:
        return {"error": "nessuna nota matchata (onset+pitch)"}

    onset_dev, dur_err = [], []
    for ref_idx, est_idx in matches:
        onset_dev.append(predicted[est_idx]["time"] - reference[ref_idx]["time"])
        dur_err.append(
            predicted[est_idx].get("duration", 0.0) - reference[ref_idx].get("duration", 0.0)
        )

    abs_onset = [abs(d) for d in onset_dev]
    within = sum(1 for d in abs_onset if d <= onset_tolerance)

    return {
        "matched_notes": len(matches),
        "onset_deviation_mean_s": round(statistics.mean(onset_dev), 4),
        "onset_deviation_median_s": round(statistics.median(onset_dev), 4),
        "onset_deviation_std_s": round(statistics.pstdev(onset_dev), 4) if len(onset_dev) > 1 else 0.0,
        "onset_abs_error_mean_s": round(statistics.mean(abs_onset), 4),
        "onset_abs_error_max_s": round(max(abs_onset), 4),
        "within_tolerance_pct": round(100 * within / len(matches), 1),
        "duration_abs_error_mean_s": round(statistics.mean(abs(e) for e in dur_err), 4),
    }


# =========================================================================
# Aggregazione + report
# =========================================================================
def compute_all_metrics(
    predicted: list[dict],
    reference: list[dict],
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
    fps: int = 100,
) -> dict:
    return {
        "totals": {"predicted": len(predicted), "reference": len(reference)},
        "note_level": note_level_metrics(predicted, reference, onset_tolerance, offset_ratio),
        "frame_level": frame_level_metrics(predicted, reference, fps),
        "timing": timing_metrics(predicted, reference, onset_tolerance),
    }


def print_metrics(m: dict) -> None:
    logger.info("")
    logger.info("=" * 64)
    logger.info("METRICHE DI ACCURATEZZA (predizione vs spartito)")
    logger.info("=" * 64)
    logger.info(f"  Note: predette={m['totals']['predicted']}  riferimento={m['totals']['reference']}")

    nl = m["note_level"]
    if "error" not in nl:
        on, off, c = nl["onset"], nl["onset_offset"], nl["counts"]
        logger.info(f"  [Note-level, tol onset {nl['onset_tolerance_s']*1000:.0f}ms]")
        logger.info(f"    onset        →  P={on['precision']:.3f}  R={on['recall']:.3f}  F1={on['f1']:.3f}")
        logger.info(f"    onset+offset →  P={off['precision']:.3f}  R={off['recall']:.3f}  F1={off['f1']:.3f}")
        logger.info(f"    overlap ratio = {nl['average_overlap_ratio']:.3f}")
        logger.info(f"    TP={c['true_positive']}  FP={c['false_positive']}  FN={c['false_negative']}")
    else:
        logger.info(f"  [Note-level] {nl['error']}")

    fl = m["frame_level"]
    if "error" not in fl:
        logger.info(f"  [Frame-level pitch, {fl['fps']} fps]")
        logger.info(f"    P={fl['precision']:.3f}  R={fl['recall']:.3f}  F1={fl['f1']:.3f}  accuracy={fl['accuracy']:.3f}")
    else:
        logger.info(f"  [Frame-level] {fl['error']}")

    tm = m["timing"]
    if "error" not in tm:
        logger.info(f"  [Tempo/durata, {tm['matched_notes']} note matchate]")
        logger.info(f"    sfasamento onset: medio={tm['onset_deviation_mean_s']:+.3f}s  "
                    f"mediano={tm['onset_deviation_median_s']:+.3f}s  std={tm['onset_deviation_std_s']:.3f}s")
        logger.info(f"    |onset err|: medio={tm['onset_abs_error_mean_s']:.3f}s  max={tm['onset_abs_error_max_s']:.3f}s")
        logger.info(f"    note entro tolleranza: {tm['within_tolerance_pct']:.1f}%")
        logger.info(f"    |durata err| medio: {tm['duration_abs_error_mean_s']:.3f}s")
    else:
        logger.info(f"  [Tempo/durata] {tm['error']}")


def save_metrics(m: dict, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    logger.info(f"  Metriche salvate:   {output_path}")
