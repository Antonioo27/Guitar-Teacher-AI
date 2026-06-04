"""
inference.py — Modulo 2: Inferenza (predizione note da audio).

Prende un file audio, lo preprocessa con la CQT e lo passa attraverso
il modello TabCNN per ottenere una sequenza di note predette.

L'output è una lista strutturata di note con timing, pitch, nome nota
e confidenza, pronta per essere confrontata con lo spartito dal Modulo 3.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import config
from .audio_processing import prepare_input_tensor
from .dataset import midi_to_note_name, tab_to_midi_pitch
from .model import TabCNN

logger = logging.getLogger(__name__)


def predict_tablature(
    model: TabCNN,
    input_tensor: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Esegue il forward pass del modello TabCNN e restituisce le predizioni
    per ogni frame e ogni corda.

    Args:
        model: Modello TabCNN in eval mode.
        input_tensor: Tensore di input (n_frames, 1, n_bins, context_frames).

    Returns:
        Tupla (predictions, confidences):
        - predictions: array (n_frames, 6) con il tasto predetto per ogni corda
                       (-1 se la corda non è suonata)
        - confidences: array (n_frames, 6) con la confidenza della predizione
    """
    n_frames = input_tensor.shape[0]
    batch_size = 256  # Processa in batch per efficienza di memoria

    all_predictions = []
    all_confidences = []

    with torch.no_grad():
        for start in range(0, n_frames, batch_size):
            end = min(start + batch_size, n_frames)
            batch = input_tensor[start:end]

            # Forward pass → lista di 6 tensori (batch, num_classes)
            outputs = model(batch)

            batch_preds = []
            batch_confs = []

            for string_output in outputs:
                # Applica softmax per ottenere probabilità
                probs = F.softmax(string_output, dim=1)

                # Classe con probabilità massima
                max_probs, max_indices = probs.max(dim=1)

                batch_preds.append(max_indices.cpu().numpy())
                batch_confs.append(max_probs.cpu().numpy())

            # Stack: (num_strings, batch) → (batch, num_strings)
            batch_preds = np.stack(batch_preds, axis=1)
            batch_confs = np.stack(batch_confs, axis=1)

            all_predictions.append(batch_preds)
            all_confidences.append(batch_confs)

    predictions = np.concatenate(all_predictions, axis=0)
    confidences = np.concatenate(all_confidences, axis=0)

    return predictions, confidences


def decode_predictions(
    predictions: np.ndarray,
    confidences: np.ndarray,
    hop_length: int = config.HOP_LENGTH,
    sr: int = config.SAMPLE_RATE,
    confidence_threshold: float = config.ONSET_THRESHOLD,
) -> list[dict]:
    """
    Decodifica le predizioni frame-by-frame della TabCNN in una sequenza
    di note discrete con onset detection.

    Implementa un semplice onset detector: una nota inizia quando il tasto
    predetto cambia rispetto al frame precedente (o quando la confidenza
    supera la soglia per la prima volta).

    Args:
        predictions: Array (n_frames, 6) con i tasti predetti.
        confidences: Array (n_frames, 6) con le confidenze.
        hop_length: Hop length usato nella CQT.
        sr: Sample rate.
        confidence_threshold: Soglia minima di confidenza.

    Returns:
        Lista di note predette, ciascuna con:
        - "time": float — tempo di onset in secondi
        - "duration": float — durata stimata in secondi
        - "pitch": int — MIDI pitch
        - "note_name": str — nome della nota
        - "string": int — indice della corda (0-5)
        - "fret": int — tasto premuto
        - "confidence": float — confidenza della predizione
    """
    n_frames, num_strings = predictions.shape
    frame_duration = hop_length / sr  # Durata di un frame in secondi

    notes = []
    # Stato corrente per ogni corda (per onset detection)
    prev_fret = [-1] * num_strings  # -1 = corda non attiva

    for t in range(n_frames):
        time_sec = t * frame_duration

        for s in range(num_strings):
            fret = int(predictions[t, s])
            conf = float(confidences[t, s])

            # Classe 0 nella TabCNN può significare "corda a vuoto" o "non suonata"
            # dipende dall'implementazione. Qui usiamo l'ultima classe come "non suonata".
            # Se fret == NUM_FRETS (20), la corda non è suonata
            if fret >= config.NUM_FRETS:
                prev_fret[s] = -1
                continue

            # Filtro per confidenza
            if conf < confidence_threshold:
                prev_fret[s] = -1
                continue

            # Onset detection: la nota inizia se il tasto è diverso dal precedente
            if fret != prev_fret[s]:
                midi_pitch = tab_to_midi_pitch(s, fret)
                notes.append({
                    "time": round(time_sec, 4),
                    "duration": round(frame_duration, 4),  # Durata minima (1 frame)
                    "pitch": midi_pitch,
                    "note_name": midi_to_note_name(midi_pitch),
                    "string": s,
                    "fret": fret,
                    "confidence": round(conf, 4),
                })
                prev_fret[s] = fret
            else:
                # Stessa nota del frame precedente: aggiorna la durata dell'ultima nota
                # su questa corda
                for note in reversed(notes):
                    if note["string"] == s and note["fret"] == fret:
                        note["duration"] = round(time_sec - note["time"] + frame_duration, 4)
                        break

    # Ordina per tempo di onset
    notes.sort(key=lambda x: (x["time"], x["string"]))
    return notes


def transcribe_audio(
    audio_path: str,
    model: TabCNN,
    device: str = "cpu",
) -> list[dict]:
    """
    Pipeline completa di trascrizione: audio → lista di note predette.

    Questa è la funzione di alto livello che combina preprocessing,
    inferenza e decodifica.

    Args:
        audio_path: Percorso al file audio.
        model: Modello TabCNN caricato in eval mode.
        device: Device PyTorch.

    Returns:
        Lista di note predette (vedi decode_predictions per il formato).
    """
    logger.info(f"Trascrizione audio: {audio_path}")

    # 1. Preprocessing: audio → tensore CQT
    input_tensor, sr, n_frames = prepare_input_tensor(audio_path, device)
    logger.info(f"Input tensor shape: {input_tensor.shape} ({n_frames} frames)")

    # 2. Inferenza: tensore → predizioni per-corda per-frame
    predictions, confidences = predict_tablature(model, input_tensor)
    logger.info(f"Predizioni shape: {predictions.shape}")

    # 3. Decodifica: predizioni frame-wise → sequenza di note
    notes = decode_predictions(predictions, confidences)
    logger.info(f"Note trascritte: {len(notes)}")

    return notes
