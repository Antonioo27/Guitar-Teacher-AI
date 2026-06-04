"""
alignment.py — Modulo 3: Sequence Alignment (La Logica di Controllo).

Allinea la sequenza di note predetta dalla TabCNN alla sequenza dello
spartito ideale usando Dynamic Time Warping (DTW).

Per ogni coppia allineata, classifica il tipo di errore:
- correct:       nota e timing corretti
- wrong_timing:  pitch corretto ma fuori tempo
- wrong_pitch:   pitch sbagliato
- missing:       nota dello spartito non suonata
- extra:         nota suonata non presente nello spartito
"""

import logging
from typing import Any

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from . import config

logger = logging.getLogger(__name__)


# =============================================================================
# Funzione di distanza personalizzata per DTW
# =============================================================================

def pitch_time_distance(
    a: tuple[float, int],
    b: tuple[float, int],
    pitch_weight: float = config.PITCH_WEIGHT,
    time_weight: float = config.TIME_WEIGHT,
) -> float:
    """
    Calcola la distanza tra due note per l'allineamento DTW.

    La distanza combina due componenti pesate:
    1. Distanza di pitch: differenza assoluta tra i MIDI pitch,
       normalizzata sulla gamma dello strumento.
    2. Distanza temporale: differenza assoluta tra i tempi di onset.

    Args:
        a: Tupla (time, pitch) della prima nota.
        b: Tupla (time, pitch) della seconda nota.

    Returns:
        Distanza scalare combinata.
    """
    time_a, pitch_a = a
    time_b, pitch_b = b

    # Distanza di pitch (normalizzata: 12 semitoni = 1 ottava)
    d_pitch = abs(pitch_a - pitch_b) / 12.0

    # Distanza temporale (in secondi)
    d_time = abs(time_a - time_b)

    return pitch_weight * d_pitch + time_weight * d_time


# =============================================================================
# Allineamento DTW
# =============================================================================

def compute_dtw_alignment(
    predicted_seq: list[dict[str, Any]],
    reference_seq: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """
    Allinea la sequenza predetta alla sequenza di riferimento usando DTW.

    Dynamic Time Warping trova l'allineamento ottimale tra due sequenze
    temporali di lunghezza diversa, tollerando variazioni di velocità.

    Args:
        predicted_seq: Sequenza di note predette dalla TabCNN.
            Ogni nota ha almeno i campi "time" e "pitch".
        reference_seq: Sequenza di note dallo spartito (ground truth).
            Ogni nota ha almeno i campi "time" e "pitch".

    Returns:
        Lista di coppie (idx_predicted, idx_reference) che rappresentano
        l'allineamento ottimale trovato dal DTW.
    """
    if not predicted_seq or not reference_seq:
        logger.warning("Una delle sequenze è vuota, impossibile allineare.")
        return []

    # Converti le sequenze in array di (time, pitch) per il DTW
    pred_array = [(n["time"], n["pitch"]) for n in predicted_seq]
    ref_array = [(n["time"], n["pitch"]) for n in reference_seq]

    # Esegui FastDTW con la metrica di distanza personalizzata
    distance, path = fastdtw(pred_array, ref_array, dist=pitch_time_distance)

    logger.info(
        f"DTW completato — Distanza totale: {distance:.2f}, "
        f"Coppie allineate: {len(path)}"
    )

    return path


# =============================================================================
# Classificazione degli errori
# =============================================================================

def classify_errors(
    path: list[tuple[int, int]],
    predicted_seq: list[dict[str, Any]],
    reference_seq: list[dict[str, Any]],
    time_tolerance: float = config.TIME_TOLERANCE,
) -> list[dict[str, Any]]:
    """
    Analizza l'allineamento DTW e classifica ogni nota in una delle
    categorie di errore definite.

    Categorie:
    - "correct": pitch corretto e timing entro la tolleranza
    - "wrong_timing": pitch corretto ma timing fuori tolleranza
    - "wrong_pitch": pitch sbagliato
    - "missing": nota dello spartito senza corrispondenza nella predizione
    - "extra": nota predetta senza corrispondenza nello spartito

    Args:
        path: Allineamento DTW (lista di coppie indice).
        predicted_seq: Sequenza predetta.
        reference_seq: Sequenza di riferimento.
        time_tolerance: Tolleranza temporale in secondi.

    Returns:
        Lista di errori, ciascuno con:
        - "time": tempo dell'evento (secondi)
        - "expected": nota attesa (o None)
        - "played": nota suonata (o None)
        - "status": tipo di errore
        - "delta_t": differenza temporale (secondi)
    """
    errors = []

    # Tieni traccia di quali note sono state coperte dall'allineamento
    matched_pred = set()
    matched_ref = set()

    for pred_idx, ref_idx in path:
        pred_note = predicted_seq[pred_idx]
        ref_note = reference_seq[ref_idx]

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
        })

        matched_pred.add(pred_idx)
        matched_ref.add(ref_idx)

    # Identifica note dello spartito non coperte (missing)
    for i, ref_note in enumerate(reference_seq):
        if i not in matched_ref:
            errors.append({
                "time": round(ref_note["time"], 3),
                "expected": ref_note.get("note_name", str(ref_note["pitch"])),
                "played": None,
                "expected_pitch": ref_note["pitch"],
                "played_pitch": None,
                "status": "missing",
                "delta_t": None,
            })

    # Identifica note predette senza corrispondenza (extra)
    for i, pred_note in enumerate(predicted_seq):
        if i not in matched_pred:
            errors.append({
                "time": round(pred_note["time"], 3),
                "expected": None,
                "played": pred_note.get("note_name", str(pred_note["pitch"])),
                "expected_pitch": None,
                "played_pitch": pred_note["pitch"],
                "status": "extra",
                "delta_t": None,
            })

    # Ordina per tempo
    errors.sort(key=lambda x: x["time"])

    return errors


# =============================================================================
# Generazione del log errori
# =============================================================================

def build_error_log(errors: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Costruisce un report strutturato degli errori per il Modulo 4 (Feedback LLM).

    Il report include statistiche aggregate e il dettaglio di ogni errore,
    nel formato atteso dal prompt engineering del Modulo 4.

    Args:
        errors: Lista di errori prodotta da classify_errors().

    Returns:
        Dizionario con:
        - "summary": statistiche aggregate
        - "errors": lista dettagliata degli errori
    """
    # Conta per categoria
    status_counts = {}
    for err in errors:
        status = err["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    total = len(errors)
    correct = status_counts.get("correct", 0)
    accuracy = (correct / total * 100) if total > 0 else 0.0

    # Filtra gli errori significativi (escludi le note corrette)
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

    logger.info(
        f"Error log — Accuracy: {accuracy:.1f}%, "
        f"Errori significativi: {len(significant_errors)}/{total}"
    )

    return {
        "summary": summary,
        "errors": significant_errors,
    }


def run_alignment(
    predicted_notes: list[dict[str, Any]],
    reference_notes: list[dict[str, Any]],
    time_tolerance: float = config.TIME_TOLERANCE,
) -> dict[str, Any]:
    """
    Funzione di alto livello che esegue l'intero Modulo 3:
    allineamento DTW + classificazione errori + report.

    Args:
        predicted_notes: Note predette dalla TabCNN (Modulo 2).
        reference_notes: Note dallo spartito di riferimento.
        time_tolerance: Tolleranza temporale in secondi.

    Returns:
        Report strutturato degli errori (vedi build_error_log).
    """
    logger.info(
        f"Alignment — Predette: {len(predicted_notes)} note, "
        f"Riferimento: {len(reference_notes)} note"
    )

    # 1. Allineamento DTW
    path = compute_dtw_alignment(predicted_notes, reference_notes)

    # 2. Classificazione errori
    errors = classify_errors(path, predicted_notes, reference_notes, time_tolerance)

    # 3. Report strutturato
    error_log = build_error_log(errors)

    return error_log
