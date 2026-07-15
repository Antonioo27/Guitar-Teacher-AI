"""
alignment.py — Modulo 3: Sequence Alignment (La Logica di Controllo).

Confronta la sequenza di note predetta dal modello di trascrizione con la
sequenza dello spartito ideale usando un matching bipartito ottimo (Hungarian
algorithm) basato sulla vicinanza temporale e sul pitch. I tempi sono confrontati
grezzi, senza correzione di offset globale né allineamento DTW.

Per ogni coppia accoppiata, classifica il tipo di errore:
- correct:       nota e timing corretti
- wrong_timing:  pitch corretto ma fuori tempo
- wrong_pitch:   pitch sbagliato
- missing:       nota dello spartito non suonata
- extra:         nota suonata non presente nello spartito
"""

import logging
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import config

logger = logging.getLogger(__name__)


# =============================================================================
# Normalizzazione delle chiavi
# =============================================================================

def _normalize_pitch(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Garantisce che ogni nota abbia la chiave "pitch".

    Le note predette da runTabCNN/inference usano "pitch", mentre quelle
    lette da un MIDI con parse_midi usano "midi_pitch". Questo helper rende
    l'allineamento indipendente dalla sorgente, senza modificare gli originali
    (restituisce copie superficiali quando serve aggiungere la chiave).
    """
    out = []
    for n in seq:
        if "pitch" not in n and "midi_pitch" in n:
            n = {**n, "pitch": n["midi_pitch"]}
        out.append(n)
    return out


# =============================================================================
# Classificazione degli errori & Matching
# =============================================================================

def classify_errors(
    predicted_seq: list[dict[str, Any]],
    reference_seq: list[dict[str, Any]],
    time_tolerance: float = config.TIME_TOLERANCE,
) -> list[dict[str, Any]]:
    """
    Esegue l'allineamento risolvendo il problema dell'accoppiamento bipartito
    (Hungarian algorithm) tra le due sequenze, guidato dalla vicinanza temporale
    tra gli onset (senza DTW né correzione di offset).
    """
    # Uniforma la chiave del pitch (pitch / midi_pitch)
    predicted_seq = _normalize_pitch(predicted_seq)
    reference_seq = _normalize_pitch(reference_seq)

    n_pred = len(predicted_seq)
    n_ref = len(reference_seq)

    if n_pred == 0:
        errors = []
        for r_idx, r_note in enumerate(reference_seq):
            errors.append({
                "time": round(r_note["time"], 3),
                "expected": r_note.get("note_name", str(r_note["pitch"])),
                "played": None,
                "expected_pitch": r_note["pitch"],
                "played_pitch": None,
                "status": "missing",
                "delta_t": None,
                "pred_idx": None,
                "ref_idx": r_idx,
            })
        return errors

    if n_ref == 0:
        errors = []
        for p_idx, p_note in enumerate(predicted_seq):
            errors.append({
                "time": round(p_note["time"], 3),
                "expected": None,
                "played": p_note.get("note_name", str(p_note["pitch"])),
                "expected_pitch": None,
                "played_pitch": p_note["pitch"],
                "status": "extra",
                "delta_t": None,
                "pred_idx": p_idx,
                "ref_idx": None,
            })
        return errors

    # 1. Matrice di costo: accoppiamo solo note vicine nel tempo
    cost_matrix = np.full((n_pred, n_ref), 1e6)
    for p_idx in range(n_pred):
        p_note = predicted_seq[p_idx]
        for r_idx in range(n_ref):
            r_note = reference_seq[r_idx]
            dt = abs(p_note["time"] - r_note["time"])

            if dt <= (time_tolerance * 2.0):
                if p_note["pitch"] == r_note["pitch"]:
                    cost = dt
                else:
                    cost = 10.0 + dt
                cost_matrix[p_idx, r_idx] = cost

    # 2. Risoluzione dell'assegnamento ottimo con l'Algoritmo Ungherese
    pred_indices, ref_indices = linear_sum_assignment(cost_matrix)

    # 3. Creiamo il report degli errori
    errors = []
    matched_pred = set()
    matched_ref = set()

    for p, r in zip(pred_indices, ref_indices):
        if cost_matrix[p, r] >= 1e5:
            continue

        pred_note = predicted_seq[p]
        ref_note = reference_seq[r]

        delta_t = pred_note["time"] - ref_note["time"]
        pitch_match = pred_note["pitch"] == ref_note["pitch"]

        if pitch_match and abs(delta_t) <= time_tolerance:
            status = "correct"
        elif pitch_match:
            status = "wrong_timing"
        else:
            status = "wrong_pitch"

        errors.append({
            "time": round(ref_note["time"], 3),
            "expected": ref_note.get("note_name", str(ref_note["pitch"])),
            "played": pred_note.get("note_name", str(pred_note["pitch"])),
            "expected_pitch": ref_note["pitch"],
            "played_pitch": pred_note["pitch"],
            "status": status,
            "delta_t": round(delta_t, 3),
            "pred_idx": int(p),
            "ref_idx": int(r),
        })

        matched_pred.add(p)
        matched_ref.add(r)

    # Note di riferimento non accoppiate -> missing
    for r in range(n_ref):
        if r not in matched_ref:
            r_note = reference_seq[r]
            errors.append({
                "time": round(r_note["time"], 3),
                "expected": r_note.get("note_name", str(r_note["pitch"])),
                "played": None,
                "expected_pitch": r_note["pitch"],
                "played_pitch": None,
                "status": "missing",
                "delta_t": None,
                "pred_idx": None,
                "ref_idx": int(r),
            })

    # Note predette non accoppiate -> extra
    for p in range(n_pred):
        if p not in matched_pred:
            p_note = predicted_seq[p]
            errors.append({
                "time": round(p_note["time"], 3),
                "expected": None,
                "played": p_note.get("note_name", str(p_note["pitch"])),
                "expected_pitch": None,
                "played_pitch": p_note["pitch"],
                "status": "extra",
                "delta_t": None,
                "pred_idx": int(p),
                "ref_idx": None,
            })

    errors.sort(key=lambda x: x["time"])
    return errors


# =============================================================================
# Generazione del log errori
# =============================================================================

def build_error_log(errors: list[dict[str, Any]], mir_eval_metrics: tuple[float, float, float] = None) -> dict[str, Any]:
    """
    Costruisce un report strutturato degli errori per il Modulo 4 (Feedback LLM).

    Il report include statistiche aggregate e il dettaglio di ogni errore.
    """
    status_counts = {}
    for err in errors:
        status = err["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    total = len(errors)
    correct = status_counts.get("correct", 0)

    if mir_eval_metrics is not None:
        P, R, F1 = mir_eval_metrics
        accuracy = F1 * 100
    else:
        accuracy = (correct / total * 100) if total > 0 else 0.0

    significant_errors = [e for e in errors if e["status"] != "correct"]

    summary = {
        "total_notes_evaluated": total,
        "correct": correct,
        "wrong_timing": status_counts.get("wrong_timing", 0),
        "wrong_pitch": status_counts.get("wrong_pitch", 0),
        "missing": status_counts.get("missing", 0),
        "extra": status_counts.get("extra", 0),
        "accuracy_percent": round(accuracy, 1),
    }

    if mir_eval_metrics is not None:
        summary["precision_percent"] = round(P * 100, 1)
        summary["recall_percent"] = round(R * 100, 1)

    logger.info(
        f"Error log — Accuracy: {accuracy:.1f}%, "
        f"Errori significativi: {len(significant_errors)}/{total}"
    )

    return {
        "summary": summary,
        "errors": significant_errors,
    }


def align_for_visualization(
    predicted_notes: list[dict[str, Any]],
    reference_notes: list[dict[str, Any]],
    time_tolerance: float = config.TIME_TOLERANCE,
) -> dict[str, Any]:
    """
    Esegue l'intero allineamento e restituisce TUTTO ciò che serve sia al
    report sia alla visualizzazione (a differenza di run_alignment, che
    restituisce solo il report filtrato).

    Returns:
        dict con:
        - "predicted":  note predette (con chiave "pitch" garantita).
        - "reference":  note di riferimento (con chiave "pitch" garantita).
        - "all_errors": classificazione COMPLETA, incl. le note "correct",
                        ogni record con "pred_idx"/"ref_idx".
        - "error_log":  report aggregato (come run_alignment).
    """
    predicted = _normalize_pitch(predicted_notes)
    reference = _normalize_pitch(reference_notes)

    # 1. Allineamento e classificazione (lista COMPLETA)
    all_errors = classify_errors(predicted, reference, time_tolerance)

    # 2. Calcolo F1 score usando mir_eval per consistenza con evaluate_synthetic.py
    import mir_eval

    if not predicted or not reference:
        P, R, F1 = 0.0, 0.0, 0.0
    else:
        ref_intervals = np.array([[n["time"], n["time"] + n.get("duration", 0.1)] for n in reference])
        ref_pitches = np.array([n["pitch"] for n in reference])

        est_intervals = np.array([[n["time"], n["time"] + n.get("duration", 0.1)] for n in predicted])
        est_pitches = np.array([n["pitch"] for n in predicted])

        P, R, F1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches, est_intervals, est_pitches,
            onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)

    # 3. Report strutturato (filtra i "correct")
    error_log = build_error_log(all_errors, mir_eval_metrics=(P, R, F1))

    return {
        "predicted": predicted,
        "reference": reference,
        "all_errors": all_errors,
        "error_log": error_log,
    }


def run_alignment(
    predicted_notes: list[dict[str, Any]],
    reference_notes: list[dict[str, Any]],
    time_tolerance: float = config.TIME_TOLERANCE,
) -> dict[str, Any]:
    """
    Funzione di alto livello che esegue l'intero Modulo 3:
    matching bipartito (Hungarian) + classificazione errori.

    Restituisce il report aggregato (summary + errori significativi).
    """
    logger.info(
        f"Alignment — Predette: {len(predicted_notes)} note, "
        f"Riferimento: {len(reference_notes)} note"
    )

    return align_for_visualization(
        predicted_notes, reference_notes, time_tolerance
    )["error_log"]
