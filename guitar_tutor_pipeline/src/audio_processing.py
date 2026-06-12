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


from amt_tools.features import CQT

def compute_cqt(
    y: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    hop_length: int = config.HOP_LENGTH,
    n_bins: int = config.N_BINS,
    bins_per_octave: int = config.BINS_PER_OCTAVE,
    fmin: float = config.FMIN,
) -> np.ndarray:
    """
    Calcola la Constant-Q Transform (CQT) del segnale audio usando amt_tools.
    Invece di implementare manualmente la logica, usiamo l'estrattore ufficiale
    per garantire una corrispondenza esatta (bit-perfect) con l'addestramento SynthTab.
    """
    extractor = CQT(
        sample_rate=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin
    )
    # process_audio esegue librosa.vqt(gamma=0) + abs + conversione db + normalizzazione / 80 + 1
    # e restituisce (1, N_BINS, T)
    feats = extractor.process_audio(y)
    return feats


def prepare_input_tensor(
    audio_path: str,
    device: str = "cpu",
) -> tuple[torch.Tensor, int, int]:
    """
    Pipeline perfettamente allineata ad amt_tools e SynthTab:
    1. Carica l'audio al SAMPLE_RATE configurato (22050 Hz per SynthTab).
    2. Calcola la CQT tramite l'estrattore ufficiale di amt_tools.
    3. Converte in tensore PyTorch e invia al device.

    Il framing locale è delegato alla funzione pre_proc() del modello TabCNN.
    """
    # 1. Caricamento audio
    y, sr = load_audio(audio_path)

    # 2. CQT (include già normalizzazione dB e unsqueeze del canale)
    # Ritorna shape: (1, n_bins, n_frames)
    feats = compute_cqt(y, sr)

    # 3. Creazione del tensore per amt_tools (Batch=1, Channels=1, Freq=192, Time=N)
    # unsqueeze(0) aggiunge la dimensione del batch
    tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)

    n_frames = feats.shape[-1]

    return tensor, sr, n_frames
