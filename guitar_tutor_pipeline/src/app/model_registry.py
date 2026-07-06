"""
model_registry.py — Registry per il caricamento dinamico dei modelli di trascrizione.

Fornisce un'interfaccia uniforme per caricare e utilizzare i modelli TabCNN e CRNN,
astraendo le differenze tra le due architetture e garantendo che l'output sia
nel formato standard richiesto dalla pipeline (lista di dizionari con campi
"time", "duration", "pitch", "note_name").

Uso:
    from .model_registry import ModelRegistry

    registry = ModelRegistry()
    model, model_name = registry.load_model("TabCNN")
    notes = registry.transcribe(audio_path, model_name)
"""

import logging
from pathlib import Path
from typing import Any, Optional

import torch

from . import config
from .dataset import midi_to_note_name

logger = logging.getLogger(__name__)


def get_available_models() -> list[dict[str, Any]]:
    """
    Restituisce l'elenco dei modelli disponibili con informazioni
    sulla disponibilità dei pesi.
    """
    models = []
    for name in config.AVAILABLE_MODELS:
        if name == "TabCNN":
            weights_path = config.WEIGHTS_PATH
            description = "TabCNN — Trascrizione per tablatura chitarra (6 corde, 20 tasti)"
        elif name == "CRNN":
            weights_path = config.CRNN_WEIGHTS_PATH
            description = "CRNN — Trascrizione note pianoforte/chitarra (88 tasti, onset regression)"
        else:
            continue

        models.append({
            "name": name,
            "description": description,
            "weights_path": str(weights_path),
            "weights_available": weights_path.exists(),
        })

    return models


def load_model(model_name: str, device: str = "cpu") -> Any:
    """
    Carica il modello specificato con i pesi pre-addestrati.

    Args:
        model_name: Nome del modello ("TabCNN" o "CRNN").
        device: Device PyTorch ("cpu" o "cuda").

    Returns:
        Modello PyTorch caricato e pronto per l'inferenza.

    Raises:
        ValueError: Se il nome del modello non è riconosciuto.
        FileNotFoundError: Se i pesi non sono disponibili.
    """
    if model_name not in config.AVAILABLE_MODELS:
        raise ValueError(
            f"Modello '{model_name}' non riconosciuto. "
            f"Modelli disponibili: {config.AVAILABLE_MODELS}"
        )

    if model_name == "TabCNN":
        from ..TabCNN_Architecture.model import load_model as load_tabcnn
        model = load_tabcnn(config.WEIGHTS_PATH, device)
        logger.info(f"Modello TabCNN caricato su device: {device}")
        return model

    elif model_name == "CRNN":
        from ..CRNN_Model.model import (
            Regress_onset_offset_frame_velocity_CRNN,
            load_maestro_checkpoint,
            load_finetuned_checkpoint,
        )
        from ..CRNN_Model.config import config as crnn_config

        model = Regress_onset_offset_frame_velocity_CRNN(
            frames_per_second=crnn_config.FRAMES_PER_SECOND,
            classes_num=crnn_config.CLASSES_NUM,
        ).to(device)

        weights_path = config.CRNN_WEIGHTS_PATH
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Pesi CRNN non trovati: {weights_path}\n"
                f"Assicurati che il file '{config.CRNN_WEIGHTS_FILENAME}' "
                f"sia presente nella directory: {config.WEIGHTS_DIR}"
            )

        # Prova prima come checkpoint fine-tuned, poi come MAESTRO
        try:
            model = load_finetuned_checkpoint(model, weights_path, device=device)
            logger.info("Checkpoint CRNN fine-tuned caricato con successo.")
        except Exception:
            model = load_maestro_checkpoint(model, weights_path, device=device)
            logger.info("Checkpoint CRNN MAESTRO caricato con successo.")

        model.eval()
        logger.info(f"Modello CRNN caricato su device: {device}")
        return model


def _crnn_notes_to_standard(raw_notes: list[tuple]) -> list[dict]:
    """
    Converte l'output del CRNN (tuple) nel formato standard della pipeline.

    Il CRNN restituisce tuple (onset_time, end_time, pitch, velocity).
    La pipeline si aspetta dizionari con campi:
        time, duration, pitch, note_name, [velocity]

    Args:
        raw_notes: Lista di tuple (onset, end, pitch, velocity) dal CRNN.

    Returns:
        Lista di dizionari nel formato standard.
    """
    standard_notes = []
    for onset, end, pitch, velocity in raw_notes:
        duration = max(end - onset, 0.01)
        standard_notes.append({
            "time": round(float(onset), 4),
            "duration": round(float(duration), 4),
            "pitch": int(pitch),
            "note_name": midi_to_note_name(int(pitch)),
            "velocity": int(velocity),
            "confidence": round(float(velocity) / 127.0, 4),
        })

    standard_notes.sort(key=lambda x: x["time"])
    return standard_notes


def transcribe(
    audio_path: str,
    model: Any,
    model_name: str,
    device: str = "cpu",
) -> list[dict]:
    """
    Trascrivi un file audio usando il modello specificato.

    Questa è l'interfaccia uniforme: indipendentemente dal modello,
    restituisce una lista di dizionari nel formato standard della pipeline.

    Args:
        audio_path: Percorso al file audio (.wav).
        model: Modello PyTorch già caricato.
        model_name: Nome del modello ("TabCNN" o "CRNN").
        device: Device PyTorch.

    Returns:
        Lista di note nel formato standard:
        [{"time", "duration", "pitch", "note_name", ...}, ...]
    """
    if model_name == "TabCNN":
        from ..TabCNN_Architecture.inference import transcribe_audio
        notes = transcribe_audio(audio_path, model, device)
        logger.info(f"[TabCNN] Trascritte {len(notes)} note da {audio_path}")
        return notes

    elif model_name == "CRNN":
        from ..CRNN_Model.inference import transcribe_full
        raw_notes = transcribe_full(model, audio_path, device, onset_thresh=0.05)
        notes = _crnn_notes_to_standard(raw_notes)
        logger.info(f"[CRNN] Trascritte {len(notes)} note da {audio_path}")
        return notes

    else:
        raise ValueError(f"Modello '{model_name}' non supportato per la trascrizione.")
