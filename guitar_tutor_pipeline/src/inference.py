"""
inference.py — Modulo 2: Inferenza (predizione note da audio).

Aggiornato per usare l'implementazione TabCNN del collega tramite amt_tools.
"""

import logging
import numpy as np
import torch

from . import config
from .audio_processing import prepare_input_tensor
from .dataset import midi_to_note_name, tab_to_midi_pitch
from .model import TabCNN

logger = logging.getLogger(__name__)


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

    Args:
        predictions: Array (n_frames, 6) con i tasti predetti (-1 indica corda muta)
        confidences: Array (n_frames, 6) con le confidenze.
        hop_length: Hop length usato nella CQT.
        sr: Sample rate.
        confidence_threshold: Soglia minima di confidenza.

    Returns:
        Lista di note predette.
    """
    n_frames, num_strings = predictions.shape
    frame_duration = hop_length / sr  # Durata di un frame in secondi

    notes = []
    prev_fret = [-1] * num_strings  # -1 = corda non attiva

    for t in range(n_frames):
        time_sec = t * frame_duration

        for s in range(num_strings):
            fret = int(predictions[t, s])
            conf = float(confidences[t, s])

            # amt_tools restituisce spesso -1 per corde vuote o classi fuori range (es. 20)
            if fret < 0 or fret >= config.NUM_FRETS:
                prev_fret[s] = -1
                continue

            # Filtro per confidenza
            if conf < confidence_threshold:
                prev_fret[s] = -1
                continue

            # Onset detection
            if fret != prev_fret[s]:
                midi_pitch = tab_to_midi_pitch(s, fret)
                notes.append({
                    "time": round(time_sec, 4),
                    "duration": round(frame_duration, 4),
                    "pitch": midi_pitch,
                    "note_name": midi_to_note_name(midi_pitch),
                    "string": s,
                    "fret": fret,
                    "confidence": round(conf, 4),
                })
                prev_fret[s] = fret
            else:
                # Stessa nota del frame precedente: aggiorna la durata
                for note in reversed(notes):
                    if note["string"] == s and note["fret"] == fret:
                        note["duration"] = round(time_sec - note["time"] + frame_duration, 4)
                        break

    notes.sort(key=lambda x: (x["time"], x["string"]))
    return notes


def transcribe_audio(
    audio_path: str,
    model: TabCNN,
    device: str = "cpu",
) -> list[dict]:
    """
    Pipeline completa di trascrizione con amt_tools:
    1. Estrazione CQT
    2. Modello pre_proc, forward, post_proc
    3. Conversione output a lista di note
    """
    logger.info(f"Trascrizione audio: {audio_path}")
    from amt_tools import tools

    # 1. Preprocessing: estrazione CQT grezza
    input_tensor, sr, n_frames_orig = prepare_input_tensor(audio_path, device)
    
    batch = {tools.KEY_FEATS: input_tensor}

    # 2. Inferenza modello amt_tools
    with torch.no_grad():
        batch = model.pre_proc(batch)
        output = model(batch[tools.KEY_FEATS])
        batch[tools.KEY_OUTPUT] = output
        risultato_finale = model.post_proc(batch)

    # 3. Estrazione matrice delle predizioni
    tablatura_stimata = risultato_finale[tools.KEY_TABLATURE]
    
    if isinstance(tablatura_stimata, torch.Tensor):
        tab_np = tablatura_stimata.cpu().numpy()
    else:
        tab_np = tablatura_stimata

    # La forma attesa da amt_tools post_proc solitamente è (Batch, Frames, Strings, 1)
    # oppure (Frames, Strings, 1). Cerchiamo di riportarla a (Frames, Strings).
    
    # Squeeze rimuove tutte le dimensioni unitarie (es. (1, 1923, 6, 1) -> (1923, 6))
    tab_np = np.squeeze(tab_np)
    
    # Se per caso l'audio è così corto da avere 1 frame (1, 6), lo squeeze diventerebbe (6,)
    if tab_np.ndim == 1:
        tab_np = tab_np.reshape(-1, 6)
    elif tab_np.ndim == 3 and tab_np.shape[1] == 6: 
        # Es: (N, 6, x)
        tab_np = tab_np[:, :, 0]
        
    predictions = tab_np
    # In assenza delle confidenze raw estratte, usiamo 1.0
    confidences = np.ones_like(predictions)

    logger.info(f"Shape predizioni estratta: {predictions.shape}")

    # 4. Decodifica: predizioni frame-wise → sequenza di note
    notes = decode_predictions(predictions, confidences)
    logger.info(f"Note trascritte: {len(notes)}")

    return notes
