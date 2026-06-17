"""
convertWawToCQT.py — Script standalone per convertire un file .wav in spettrogramma CQT.

Questo script esegue la prima parte della pipeline TabCNN:
    .wav  →  forma d'onda  →  CQT complessa  →  magnitude  →  dB  →  normalizzazione [0,1]

Output salvati:
    - <nome>_cqt.npy        → matrice NumPy (192, T) pronta per il modello
    - <nome>_cqt.png        → immagine dello spettrogramma per visualizzazione

Uso:
    python -m guitar_tutor_pipeline.src.convertWawToCQT path/to/file.wav [--output_dir path/to/output]
"""

import sys
import logging
import argparse
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt

# =========================================================================
# Configurazione del logger — mostra tutto ciò che succede passo per passo
# =========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =========================================================================
# Parametri CQT — DEVONO essere identici a quelli usati dal modello TabCNN
# (copiati da config.py per rendere lo script indipendente)
# =========================================================================
SAMPLE_RATE = 22050          # Frequenza di campionamento: 22050 "foto" del suono al secondo
HOP_LENGTH = 512             # Ogni quanti campioni calcoliamo un frame CQT
N_BINS = 192                 # Numero di bande di frequenza (192 = 8 ottave × 24 bin/ottava)
BINS_PER_OCTAVE = 24         # Risoluzione: 24 bin per ottava = quarti di tono
FMIN = 32.70319566257483     # Frequenza minima: C1 (la nota più bassa analizzata)


# =========================================================================
# FASE 1 — Caricamento del file audio
# =========================================================================
def load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """
    Carica un file .wav e lo converte in un array di numeri.

    Cosa fa librosa.load():
        - Legge il file audio dal disco
        - Lo ricampiona a SAMPLE_RATE (22050 Hz) se necessario
        - Lo converte in mono (un solo canale) se era stereo
        - Restituisce un array 1D di float32 nell'intervallo [-1.0, 1.0]

    Ogni numero nell'array rappresenta la posizione della membrana del
    microfono in un dato istante. 22050 numeri = 1 secondo di audio.

    Returns:
        y:  array numpy (N,) con i campioni audio
        sr: sample rate effettivo (sempre 22050)
    """
    logger.info("=" * 70)
    logger.info("FASE 1 — Caricamento audio")
    logger.info("=" * 70)

    # Verifica che il file esista
    path = Path(audio_path)
    if not path.exists():
        logger.error(f"File non trovato: {audio_path}")
        sys.exit(1)

    # Dimensione del file su disco
    file_size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"  File:        {path.name}")
    logger.info(f"  Percorso:    {path.absolute()}")
    logger.info(f"  Dimensione:  {file_size_mb:.2f} MB")

    # Caricamento effettivo con librosa
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    # Log dettagliati sul risultato
    durata_sec = len(y) / sr
    logger.info(f"  Tipo dato:   {type(y).__name__} (dtype={y.dtype})")
    logger.info(f"  Shape:       {y.shape}  →  {len(y):,} campioni")
    logger.info(f"  Sample rate: {sr} Hz")
    logger.info(f"  Durata:      {durata_sec:.2f} secondi")
    logger.info(f"  Valori:      min={y.min():.4f}  max={y.max():.4f}  media={y.mean():.6f}")
    logger.info(f"  Silenzio?    {'Sì (segnale molto debole)' if np.abs(y).max() < 0.01 else 'No'}")

    return y, sr


# =========================================================================
# FASE 2 — Calcolo della CQT (Constant-Q Transform)
# =========================================================================
def compute_cqt(
    y: np.ndarray,
    sr: int,
    hop_length: int = HOP_LENGTH,
    n_bins: int = N_BINS,
    bins_per_octave: int = BINS_PER_OCTAVE,
    fmin: float = FMIN,
) -> np.ndarray:
    """
    Trasforma l'array audio in uno spettrogramma CQT.

    La CQT scompone il suono nelle sue frequenze componenti, come un
    prisma scompone la luce nei colori. Il risultato è una matrice 2D
    dove:
        - Righe  = bande di frequenza (da C1=32.7Hz verso l'acuto)
        - Colonne = istanti di tempo (un frame ogni HOP_LENGTH campioni)

    La CQT è preferita alla FFT per la musica perché le sue bande sono
    distribuite logaritmicamente, come le note su una tastiera.

    Parametri usati:
        - n_bins=192:         192 bande di frequenza
        - bins_per_octave=24: 24 bande per ottava (risoluzione quarti di tono)
        - hop_length=512:     un frame ogni 512 campioni (~23ms)
        - fmin=32.7 Hz:       nota più bassa = C1

    Returns:
        cqt: matrice complessa (192, T) dove T = numero di frame temporali
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("FASE 2 — Calcolo CQT (Constant-Q Transform)")
    logger.info("=" * 70)

    # Log dei parametri usati
    frame_duration_ms = (hop_length / sr) * 1000
    n_frames_attesi = len(y) // hop_length + 1
    logger.info(f"  Parametri CQT:")
    logger.info(f"    n_bins:          {n_bins} bande di frequenza")
    logger.info(f"    bins_per_octave: {bins_per_octave}")
    logger.info(f"    hop_length:      {hop_length} campioni ({frame_duration_ms:.1f} ms per frame)")
    logger.info(f"    fmin:            {fmin:.2f} Hz")
    logger.info(f"    n_ottave:        {n_bins // bins_per_octave}")
    logger.info(f"    frame attesi:    ~{n_frames_attesi}")

    # Calcolo della CQT
    logger.info("  Calcolo in corso...")
    cqt = librosa.cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
    )

    # Log del risultato
    logger.info(f"  Risultato CQT:")
    logger.info(f"    Tipo:    {type(cqt).__name__} (dtype={cqt.dtype})")
    logger.info(f"    Shape:   {cqt.shape}  →  ({cqt.shape[0]} frequenze × {cqt.shape[1]} frame)")
    logger.info(f"    Complesso: Sì — ogni cella ha ampiezza + fase")
    logger.info(f"    Durata coperta: {cqt.shape[1] * hop_length / sr:.2f} secondi")

    return cqt


# =========================================================================
# FASE 3 — Estrazione della Magnitude (ampiezza)
# =========================================================================
def extract_magnitude(cqt: np.ndarray) -> np.ndarray:
    """
    Prende il valore assoluto della CQT complessa per ottenere solo
    l'ampiezza (quanto è forte ogni frequenza), scartando la fase
    (in che punto dell'oscillazione si trova).

    La fase non serve per riconoscere le note — ci interessa solo
    QUANTO FORTE suona ogni frequenza, non dove si trova nell'onda.

    Returns:
        cqt_mag: matrice reale (192, T) con valori >= 0
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("FASE 3 — Estrazione magnitude (ampiezza)")
    logger.info("=" * 70)

    cqt_mag = np.abs(cqt)

    logger.info(f"  Shape:    {cqt_mag.shape} (invariata)")
    logger.info(f"  Dtype:    {cqt_mag.dtype}  (era {cqt.dtype}, ora numeri reali)")
    logger.info(f"  Valori:   min={cqt_mag.min():.6f}  max={cqt_mag.max():.6f}")
    logger.info(f"  Media:    {cqt_mag.mean():.6f}")

    # Statistiche per capire quanta "energia" c'è nello spettrogramma
    silenzio_pct = np.sum(cqt_mag < 1e-6) / cqt_mag.size * 100
    logger.info(f"  Celle ~silenziose (<1e-6): {silenzio_pct:.1f}% del totale")

    return cqt_mag


# =========================================================================
# FASE 4 — Normalizzazione in Decibel
# =========================================================================
def normalize_to_db(cqt_mag: np.ndarray) -> np.ndarray:
    """
    Converte l'ampiezza in scala decibel e normalizza a [0, 1].

    Perché i decibel?
        L'orecchio umano percepisce il volume in modo logaritmico:
        un suono con ampiezza doppia NON sembra "il doppio più forte".
        I decibel sono una scala logaritmica che rispecchia questa percezione.

    Perché normalizzare a [0, 1]?
        Le reti neurali lavorano meglio con valori piccoli e uniformi.
        Valori troppo grandi o troppo variabili rendono instabile
        l'apprendimento.

    Passaggi:
        1. amplitude_to_db():  ampiezza → dB (range tipico: [-80, 0])
           - 0 dB   = il valore più forte (il massimo)
           - -80 dB  = quasi silenzio (il minimo consentito)
        2. (dB / 80) + 1:     dB → [0, 1]
           - -80 dB → (-80/80) + 1 = 0.0
           -   0 dB → (  0/80) + 1 = 1.0

    Returns:
        cqt_norm: matrice (192, T) con valori in [0.0, 1.0]
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("FASE 4 — Normalizzazione in decibel")
    logger.info("=" * 70)

    # Passo 1: conversione in dB
    cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)
    logger.info(f"  Dopo amplitude_to_db():")
    logger.info(f"    Shape:   {cqt_db.shape}")
    logger.info(f"    Range:   [{cqt_db.min():.2f}, {cqt_db.max():.2f}] dB")
    logger.info(f"    0 dB = suono più forte, -80 dB = quasi silenzio")

    # Passo 2: normalizzazione a [0, 1]
    cqt_norm = (cqt_db / 80.0) + 1.0
    logger.info(f"  Dopo normalizzazione (dB/80 + 1):")
    logger.info(f"    Shape:   {cqt_norm.shape}")
    logger.info(f"    Range:   [{cqt_norm.min():.4f}, {cqt_norm.max():.4f}]")
    logger.info(f"    Dtype:   {cqt_norm.dtype}")

    return cqt_norm


# =========================================================================
# Salvataggio su disco
# =========================================================================
def save_results(cqt_norm: np.ndarray, audio_path: str, output_dir: str) -> None:
    """
    Salva la matrice CQT normalizzata in due formati:
        1. .npy  — formato NumPy binario (ricaricabile con np.load())
        2. .png  — immagine dello spettrogramma per visualizzazione

    Args:
        cqt_norm:   matrice (192, T) normalizzata in [0, 1]
        audio_path: percorso del file audio originale (per derivare il nome)
        output_dir: cartella dove salvare i risultati
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("SALVATAGGIO RISULTATI")
    logger.info("=" * 70)

    # Crea la cartella di output se non esiste
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Nome base derivato dal file audio
    stem = Path(audio_path).stem  # es: "00_CMaj_120" da "00_CMaj_120.wav"

    # --- Salvataggio .npy (matrice numerica) ---
    npy_path = out_path / f"{stem}_cqt.npy"
    np.save(npy_path, cqt_norm)
    size_kb = npy_path.stat().st_size / 1024
    logger.info(f"  Matrice NumPy salvata: {npy_path}")
    logger.info(f"    Shape: {cqt_norm.shape}  |  Dimensione: {size_kb:.1f} KB")
    logger.info(f"    Per ricaricare: cqt = np.load('{npy_path}')")

    # --- Salvataggio .png (immagine spettrogramma) ---
    png_path = out_path / f"{stem}_cqt.png"

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    img = ax.imshow(
        cqt_norm,
        aspect="auto",
        origin="lower",          # frequenze basse in basso
        cmap="magma",            # colormap: nero=silenzio, giallo=forte
        vmin=0.0,
        vmax=1.0,
    )

    # Etichette degli assi
    # Asse X: converti frame in secondi
    n_frames = cqt_norm.shape[1]
    frame_dur = HOP_LENGTH / SAMPLE_RATE
    tick_positions = np.linspace(0, n_frames - 1, num=10, dtype=int)
    tick_labels = [f"{t * frame_dur:.1f}" for t in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Tempo (secondi)")

    # Asse Y: mostra alcune note di riferimento
    ax.set_ylabel("Banda di frequenza (bin CQT)")
    ax.set_title(f"Spettrogramma CQT — {Path(audio_path).name}")

    plt.colorbar(img, ax=ax, label="Ampiezza normalizzata [0, 1]")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)

    logger.info(f"  Immagine PNG salvata:  {png_path}")
    logger.info(f"    Apri con: xdg-open '{png_path}'")


# =========================================================================
# MAIN — Entry point per esecuzione standalone
# =========================================================================
def main():
    """
    Entry point dello script.

    Uso:
        python -m guitar_tutor_pipeline.src.convertWawToCQT file.wav
        python -m guitar_tutor_pipeline.src.convertWawToCQT file.wav --output_dir output/
    """
    # Parsing degli argomenti da riga di comando
    parser = argparse.ArgumentParser(
        description="Converte un file .wav nello spettrogramma CQT usato da TabCNN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Esempio:\n"
            "  python -m guitar_tutor_pipeline.src.convertWawToCQT audio.wav\n"
            "  python -m guitar_tutor_pipeline.src.convertWawToCQT audio.wav --output_dir risultati/\n"
        ),
    )
    parser.add_argument("audio_path", type=str, help="Percorso al file audio .wav da convertire")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_cqt",
        help="Cartella dove salvare i risultati (default: output_cqt/)",
    )
    parser.add_argument("--hop_length", type=int, default=None,
                        help=f"Campioni tra frame CQT (default: {HOP_LENGTH})")
    parser.add_argument("--bins_per_octave", type=int, default=None,
                        help=f"Bin per ottava (default: {BINS_PER_OCTAVE})")
    parser.add_argument("--n_bins", type=int, default=None,
                        help=f"Numero totale di bin (default: {N_BINS})")
    parser.add_argument("--fmin", type=float, default=None,
                        help=f"Frequenza minima Hz (default: {FMIN:.2f} = C1)")
    args = parser.parse_args()

    # Sovrascrive i parametri globali se passati via CLI
    hop_length      = args.hop_length      if args.hop_length      is not None else HOP_LENGTH
    bins_per_octave = args.bins_per_octave if args.bins_per_octave is not None else BINS_PER_OCTAVE
    n_bins          = args.n_bins          if args.n_bins          is not None else N_BINS
    fmin            = args.fmin            if args.fmin            is not None else FMIN

    logger.info("🎸 Conversione WAV → CQT per TabCNN")
    logger.info(f"  Input:           {args.audio_path}")
    logger.info(f"  Output:          {args.output_dir}/")
    logger.info(f"  hop_length:      {hop_length}")
    logger.info(f"  bins_per_octave: {bins_per_octave}")
    logger.info(f"  n_bins:          {n_bins}")
    logger.info(f"  fmin:            {fmin:.2f} Hz")
    logger.info("")

    # ── Esecuzione della pipeline ──────────────────────────────────────
    y, sr = load_audio(args.audio_path)

    cqt = compute_cqt(y, sr, hop_length=hop_length, n_bins=n_bins,
                      bins_per_octave=bins_per_octave, fmin=fmin)

    cqt_mag = extract_magnitude(cqt)

    cqt_norm = normalize_to_db(cqt_mag)

    # Salvataggio su disco (.npy + .png)
    save_results(cqt_norm, args.audio_path, args.output_dir)

    # ── Riepilogo finale ──────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ CONVERSIONE COMPLETATA")
    logger.info("=" * 70)
    logger.info(f"  Input:           {args.audio_path}")
    logger.info(f"  Durata audio:    {len(y) / sr:.2f} s")
    logger.info(f"  Matrice CQT:     {cqt_norm.shape}  (frequenze × frame)")
    logger.info(f"  Range valori:    [{cqt_norm.min():.4f}, {cqt_norm.max():.4f}]")
    logger.info(f"  File salvati in: {args.output_dir}/")


if __name__ == "__main__":
    main()
