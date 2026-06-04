"""
audio_processing.py — Modulo 1: Data Ingestion & Preprocessing (L'Udito).

Trasforma i dati audio grezzi (file .wav) in tensori CQT pronti
per l'input alla rete neurale TabCNN. (Aggiornato per supportare amt_tools)
"""

import numpy as np
import librosa
import torch

from . import config


def load_audio(
    audio_path: str,
    sr: int = config.SAMPLE_RATE,
    duration: float | None = config.AUDIO_DURATION,
) -> tuple[np.ndarray, int]:
    """
    Carica un file audio e lo ricampiona alla frequenza desiderata.
    """
    y, sr_out = librosa.load(audio_path, sr=sr, mono=True, duration=duration)
    return y, sr_out


def compute_cqt(
    y: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    hop_length: int = config.HOP_LENGTH,
    n_bins: int = config.N_BINS,
    bins_per_octave: int = config.BINS_PER_OCTAVE,
    fmin: float = config.FMIN,
) -> np.ndarray:
    """
    Calcola la Constant-Q Transform (CQT) del segnale audio.
    """
    cqt = librosa.cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
    )
    return cqt


def prepare_input_tensor(
    audio_path: str,
    device: str = "cpu",
) -> tuple[torch.Tensor, int, int]:
    """
    Pipeline compatibile con amt_tools:
    1. Carica l'audio
    2. Calcola la CQT
    3. Converte in decibel e normalizza (come in amt_tools VQT)
    4. Crea il tensore (1, 1, n_bins, n_frames) 

    Il framing locale è delegato alla funzione pre_proc() del modello TabCNN.
    """
    # 1. Caricamento
    y, sr = load_audio(audio_path)

    # 2. CQT
    cqt = compute_cqt(y, sr)
    cqt_mag = np.abs(cqt)

    # 3. Normalizzazione come in amt_tools.features.VQT (decibels=True)
    cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)
    cqt_norm = (cqt_db / 80.0) + 1.0

    # 4. Creazione del tensore per amt_tools (Batch=1, Channels=1, Freq=192, Time=N)
    tensor = torch.tensor(cqt_norm, dtype=torch.float32).unsqueeze(0).to(device)

    n_frames = cqt_norm.shape[1]

    return tensor, sr, n_frames
