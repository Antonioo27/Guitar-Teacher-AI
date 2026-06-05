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
from scipy.optimize import linear_sum_assignment

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
# Classificazione degli errori & Matching
# =============================================================================

def estimate_global_offset(
    predicted_seq: list[dict[str, Any]],
    reference_seq: list[dict[str, Any]],
) -> float:
    """
    Stima l'offset temporale globale (latenza o silenzio iniziale) tra la
    sequenza predetta e quella di riferimento confrontando i tempi di onset
    delle note con lo stesso pitch.
    """
    if not predicted_seq or not reference_seq:
        return 0.0

    # Consideriamo una finestra dei primi 40 elementi di ciascuna sequenza
    diffs = []
    for p in predicted_seq[:40]:
        for r in reference_seq[:40]:
            if p["pitch"] == r["pitch"]:
                diffs.append(p["time"] - r["time"])

    if not diffs:
        # Se non ci sono note con pitch corrispondenti all'inizio,
        # usiamo la differenza tra i primissimi onset assoluti.
        return predicted_seq[0]["time"] - reference_seq[0]["time"]

    # Cerchiamo la differenza più frequente (il picco del cluster)
    best_offset = 0.0
    max_matches = -1
    
    # Raggruppiamo i valori simili entro una finestra di 150 ms
    for d in diffs:
        matches = sum(1 for x in diffs if abs(x - d) <= 0.15)
        if matches > max_matches:
            max_matches = matches
            best_offset = d

    logger.info(f"Stima dell'offset globale completata: {best_offset:.3f} s (trovati {max_matches} match coerenti)")
    return best_offset


def classify_errors(
    path: list[tuple[int, int]],
    predicted_seq: list[dict[str, Any]],
    reference_seq: list[dict[str, Any]],
    time_tolerance: float = config.TIME_TOLERANCE,
) -> list[dict[str, Any]]:
    """
    Esegue l'allineamento di dettaglio risolvendo il problema dell'accoppiamento
    bipartito (Hungarian algorithm) tra le due sequenze, guidato dalle adiacenze
    identificate dal DTW o dalla vicinanza temporale.
    """
    n_pred = len(predicted_seq)
    n_ref = len(reference_seq)
    
    if n_pred == 0:
        errors = []
        for r_note in reference_seq:
            errors.append({
                "time": round(r_note["time"], 3),
                "expected": r_note.get("note_name", str(r_note["pitch"])),
                "played": None,
                "expected_pitch": r_note["pitch"],
                "played_pitch": None,
                "status": "missing",
                "delta_t": None,
            })
        return errors

    if n_ref == 0:
        errors = []
        for p_note in predicted_seq:
            time_to_display = p_note.get("time_original", p_note["time"])
            errors.append({
                "time": round(time_to_display, 3),
                "expected": None,
                "played": p_note.get("note_name", str(p_note["pitch"])),
                "expected_pitch": None,
                "played_pitch": p_note["pitch"],
                "status": "extra",
                "delta_t": None,
            })
        return errors

    # 1. Costruiamo il set di coppie candidate allineate dal DTW
    dtw_candidates = set(path)
    
    # 2. Inizializziamo la matrice di costo a un valore elevato
    cost_matrix = np.full((n_pred, n_ref), 1e6)
    
    # Calcoliamo il costo per coppie candidate o vicine nel tempo
    for p_idx in range(n_pred):
        p_note = predicted_seq[p_idx]
        for r_idx in range(n_ref):
            r_note = reference_seq[r_idx]
            dt = abs(p_note["time"] - r_note["time"])
            
            is_dtw_candidate = (p_idx, r_idx) in dtw_candidates
            is_locally_close = dt <= (time_tolerance * 2.0)
            
            if is_dtw_candidate or is_locally_close:
                if p_note["pitch"] == r_note["pitch"]:
                    cost = dt
                else:
                    cost = 10.0 + dt
                
                cost_matrix[p_idx, r_idx] = cost

    # 3. Risoluzione dell'assegnamento ottimo con l'Algoritmo Ungherese
    pred_indices, ref_indices = linear_sum_assignment(cost_matrix)
    
    # 4. Creiamo il report degli errori
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
            })

    # Note predette non accoppiate -> extra
    for p in range(n_pred):
        if p not in matched_pred:
            p_note = predicted_seq[p]
            time_to_display = p_note.get("time_original", p_note["time"])
            errors.append({
                "time": round(time_to_display, 3),
                "expected": None,
                "played": p_note.get("note_name", str(p_note["pitch"])),
                "expected_pitch": None,
                "played_pitch": p_note["pitch"],
                "status": "extra",
                "delta_t": None,
            })

    errors.sort(key=lambda x: x["time"])
    return errors


# =============================================================================
# Generazione del log errori
# =============================================================================

def build_error_log(errors: list[dict[str, Any]]) -> dict[str, Any]:
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
    stima offset + allineamento DTW + Bipartite Matching + classificazione errori.
    """
    logger.info(
        f"Alignment — Predette: {len(predicted_notes)} note, "
        f"Riferimento: {len(reference_notes)} note"
    )

    # 1. Stima e correzione dell'offset globale
    global_offset = estimate_global_offset(predicted_notes, reference_notes)
    
    shifted_predicted = []
    for note in predicted_notes:
        shifted_note = note.copy()
        shifted_note["time_original"] = note["time"]
        shifted_note["time"] = max(0.0, note["time"] - global_offset)
        shifted_predicted.append(shifted_note)

    # 2. Allineamento grossolano DTW sulla sequenza shiftata
    path = compute_dtw_alignment(shifted_predicted, reference_notes)

    # 3. Allineamento di dettaglio e classificazione con Algoritmo Ungherese
    errors = classify_errors(path, shifted_predicted, reference_notes, time_tolerance)

    # 4. Report strutturato
    error_log = build_error_log(errors)
    
    # Aggiungiamo l'offset stimato al summary per trasparenza
    error_log["summary"]["estimated_global_offset_sec"] = round(global_offset, 3)

    return error_log
