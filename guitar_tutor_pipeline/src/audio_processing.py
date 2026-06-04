"""
audio_processing.py — Modulo 1: Data Ingestion & Preprocessing (L'Udito).

Trasforma i dati audio grezzi (file .wav) in tensori CQT pronti
per l'input alla rete neurale TabCNN.

La Constant-Q Transform (CQT) è preferita al Mel-Spectrogram perché
allinea geometricamente le frequenze alle note musicali, con risoluzione
logaritmica che corrisponde alla scala cromatica.
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

    Args:
        audio_path: Percorso al file audio (.wav, .mp3, .flac, ecc.)
        sr: Frequenza di campionamento target.
        duration: Durata massima in secondi (None = intero file).

    Returns:
        Tupla (segnale_audio, sample_rate).
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

    La CQT produce una rappresentazione tempo-frequenza dove ogni bin
    corrisponde a un intervallo musicale fisso (es. semitono), ideale
    per compiti di trascrizione musicale.

    Args:
        y: Segnale audio.
        sr: Frequenza di campionamento.
        hop_length: Numero di campioni tra frame successivi.
        n_bins: Numero totale di bin frequenziali.
        bins_per_octave: Risoluzione frequenziale per ottava.
        fmin: Frequenza minima in Hz.

    Returns:
        Matrice CQT complessa di forma (n_bins, n_frames).
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


def normalize_cqt(cqt: np.ndarray) -> np.ndarray:
    """
    Converte la CQT complessa in ampiezza in scala dB e normalizza.

    La normalizzazione porta i valori nell'intervallo [0, 1] per
    facilitare l'addestramento della rete neurale.

    Args:
        cqt: Matrice CQT complessa (n_bins, n_frames).

    Returns:
        Matrice CQT normalizzata in dB (n_bins, n_frames), valori in [0, 1].
    """
    # Ampiezza → scala logaritmica (dB)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    # Normalizza in [0, 1]: il valore massimo (0 dB) diventa 1,
    # il minimo (~-80 dB) diventa 0
    cqt_norm = (cqt_db - cqt_db.min()) / (cqt_db.max() - cqt_db.min() + 1e-8)

    return cqt_norm


def extract_cqt_frames(
    cqt_norm: np.ndarray,
    context_size: int = config.CONTEXT_FRAMES,
) -> np.ndarray:
    """
    Estrae frame con contesto temporale per l'input della TabCNN.

    Per ogni frame temporale t, crea una finestra di contesto
    [t - context_size//2, t + context_size//2] che cattura
    l'informazione temporale locale.

    Args:
        cqt_norm: CQT normalizzata (n_bins, n_frames).
        context_size: Numero di frame nel contesto temporale.

    Returns:
        Array di shape (n_frames, 1, n_bins, context_size) —
        batch di "immagini" CQT locali pronte per la CNN.
    """
    n_bins, n_frames = cqt_norm.shape
    half_ctx = context_size // 2

    # Padding temporale con zeri ai bordi
    padded = np.pad(cqt_norm, ((0, 0), (half_ctx, half_ctx)), mode="constant")

    frames = []
    for t in range(n_frames):
        # Finestra locale centrata su t
        frame = padded[:, t : t + context_size]  # (n_bins, context_size)
        frames.append(frame)

    # (n_frames, n_bins, context_size) → (n_frames, 1, n_bins, context_size)
    frames = np.array(frames)[:, np.newaxis, :, :]
    return frames


def prepare_input_tensor(
    audio_path: str,
    device: str = "cpu",
) -> tuple[torch.Tensor, int, int]:
    """
    Pipeline completa: audio → tensore CQT pronto per la TabCNN.

    Esegue in sequenza:
    1. Caricamento audio
    2. Calcolo CQT
    3. Normalizzazione in dB
    4. Estrazione frame con contesto temporale
    5. Conversione in tensore PyTorch

    Args:
        audio_path: Percorso al file audio.
        device: Device PyTorch ("cpu" o "cuda").

    Returns:
        Tupla (tensor, sample_rate, n_frames_originali) dove tensor ha
        shape (n_frames, 1, n_bins, context_size).
    """
    # 1. Caricamento
    y, sr = load_audio(audio_path)

    # 2. CQT
    cqt = compute_cqt(y, sr)

    # 3. Normalizzazione
    cqt_norm = normalize_cqt(cqt)

    # 4. Estrazione frame con contesto
    frames = extract_cqt_frames(cqt_norm)

    # 5. Conversione a tensore PyTorch
    tensor = torch.tensor(frames, dtype=torch.float32).to(device)

    n_frames = cqt_norm.shape[1]

    return tensor, sr, n_frames
