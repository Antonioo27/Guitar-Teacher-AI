"""
runTabCNN.py — Script standalone per trascrivere una matrice CQT (.npy) con TabCNN.

Questo script esegue l'inferenza del modello definito in model.py a partire
NON dall'audio grezzo, ma dalla matrice CQT già normalizzata in decibel
prodotta da convertWawToCQT.py:

    <nome>_cqt.npy  →  tensore  →  TabCNN  →  predizioni frame-by-frame  →  note

In questo modo possiamo procedere per step: prima si genera la CQT con
convertWawToCQT.py, poi la si dà in pasto al modello con questo script,
variando eventualmente i parametri della CQT a monte.

La matrice .npy attesa ha:
    - shape (N_BINS, T)  →  (192, n_frame)
    - valori normalizzati in [0, 1]  (= (dB / 80) + 1)

Output salvati (opzionali, con --output_dir):
    - <nome>_notes.json   → lista di note in formato JSON
    - <nome>_notes.csv    → stessa lista in formato tabellare

Uso:
    python -m guitar_tutor_pipeline.src.runTabCNN output_cqt/00_CMaj_120_cqt.npy
    python -m guitar_tutor_pipeline.src.runTabCNN matrice.npy --output_dir output_notes/
    python -m guitar_tutor_pipeline.src.runTabCNN matrice.npy --device cuda
"""

import sys
import csv
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import torch

from . import config
from .model import load_model
from .inference import decode_predictions

# =========================================================================
# Configurazione del logger — coerente con convertWawToCQT.py
# =========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =========================================================================
# FASE A — Caricamento della matrice CQT dal file .npy
# =========================================================================
def load_cqt_matrix(npy_path: str) -> np.ndarray:
    """
    Carica la matrice CQT normalizzata da un file .npy su disco.

    Si aspetta una matrice 2D (N_BINS, T) con valori in [0, 1], cioè
    esattamente l'output di convertWawToCQT.py.

    Returns:
        cqt_norm: array numpy (192, T) di tipo float32
    """
    logger.info("=" * 70)
    logger.info("FASE A — Caricamento matrice CQT (.npy)")
    logger.info("=" * 70)

    path = Path(npy_path)
    if not path.exists():
        logger.error(f"File .npy non trovato: {npy_path}")
        sys.exit(1)

    cqt_norm = np.load(path)

    # Validazione della forma: ci aspettiamo (N_BINS, T)
    if cqt_norm.ndim != 2:
        logger.error(
            f"Matrice con {cqt_norm.ndim} dimensioni: attesa una matrice 2D (n_bins, n_frame)."
        )
        sys.exit(1)

    if cqt_norm.shape[0] != config.N_BINS:
        logger.warning(
            f"  Attenzione: la matrice ha {cqt_norm.shape[0]} bin, "
            f"ma il modello ne attende {config.N_BINS}. "
            f"Probabile mismatch dei parametri CQT (n_bins)."
        )

    # Assicura il tipo corretto per PyTorch
    cqt_norm = cqt_norm.astype(np.float32)

    logger.info(f"  File:        {path.name}")
    logger.info(f"  Shape:       {cqt_norm.shape}  →  ({cqt_norm.shape[0]} bin × {cqt_norm.shape[1]} frame)")
    logger.info(f"  Dtype:       {cqt_norm.dtype}")
    logger.info(f"  Range:       [{cqt_norm.min():.4f}, {cqt_norm.max():.4f}]")
    if cqt_norm.min() < -0.01 or cqt_norm.max() > 1.01:
        logger.warning("  I valori escono da [0, 1]: la matrice potrebbe non essere normalizzata.")

    return cqt_norm


# =========================================================================
# FASE B — Costruzione del tensore di input per il modello
# =========================================================================
def build_input_tensor(cqt_norm: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """
    Converte la matrice CQT (192, T) nel tensore atteso da TabCNN.

    Il modello (tramite pre_proc) si aspetta un batch con la feature map
    nella forma (Batch=1, Freq=192, Time=T). Il framing locale a finestre
    di CONTEXT_FRAMES è poi gestito internamente da pre_proc().

    Returns:
        tensor: torch.Tensor di shape (1, 192, T) sul device richiesto
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("FASE B — Costruzione tensore di input")
    logger.info("=" * 70)

    # (192, T) → (1, 192, T): aggiunge la dimensione di batch
    tensor = torch.tensor(cqt_norm, dtype=torch.float32).unsqueeze(0).to(device)

    logger.info(f"  Tensor shape: {tuple(tensor.shape)}  (Batch=1, Freq, Time)")
    logger.info(f"  Device:       {tensor.device}")

    return tensor


# =========================================================================
# FASE C — Esecuzione del modello (forward + estrazione confidenze)
# =========================================================================
def run_model(model, input_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    Esegue l'inferenza del modello TabCNN sul tensore di input.

    Passaggi (replicati qui invece di usare inference.transcribe_audio):
        1. pre_proc   — framing locale a finestre di contesto
        2. forward    — produce la distribuzione softmax per (frame, corda, classe)
        3. confidenze — max sulla distribuzione softmax = quanto è sicuro il modello
        4. post_proc  — argmax → tasto predetto per ogni (frame, corda)

    Returns:
        predictions: matrice (T, 6) con il fret predetto per frame e corda
        confidences: matrice (T, 6) con la confidenza [0,1] di ogni predizione
    """
    from amt_tools import tools

    logger.info("")
    logger.info("=" * 70)
    logger.info("FASE C — Inferenza TabCNN")
    logger.info("=" * 70)

    batch = {tools.KEY_FEATS: input_tensor}

    with torch.no_grad():
        # 1. Pre-processing interno del modello (framing locale)
        batch = model.pre_proc(batch)

        # 2. Forward → distribuzione softmax grezza
        output = model(batch[tools.KEY_FEATS])
        batch[tools.KEY_OUTPUT] = output

        # 3. Estrazione confidenze
        #    ATTENZIONE: output[KEY_TABLATURE] sono LOGIT grezzi (output di un
        #    nn.Linear), NON probabilità. La softmax la applica il modello solo
        #    dentro finalize_output. Quindi qui dobbiamo applicarla noi sulle
        #    21 classi per ottenere una confidenza reale in [0, 1].
        raw_logits = output[tools.KEY_TABLATURE]
        raw_np = (
            raw_logits.cpu().numpy()
            if isinstance(raw_logits, torch.Tensor)
            else np.array(raw_logits)
        )

        # Rimuove le dimensioni fittizie di lunghezza 1 (es. (705, 1, 126) → (705, 126))
        raw_np = np.squeeze(raw_np)

        # Porta i logit alla forma (T, 6, 21): un gruppo softmax per corda
        if raw_np.ndim == 2:
            # Caso "schiacciato" (T, 126) → separa gruppi (6) × classi (21)
            T_frames = raw_np.shape[0]
            logits = raw_np.reshape(T_frames, 6, -1)
        elif raw_np.ndim == 3:
            # Caso già separato (T, 6, 21)
            logits = raw_np
        else:
            logger.warning(f"Shape logit inattesa {raw_np.shape}: confidenze = 1.0")
            logits = None

        if logits is not None:
            # Softmax numericamente stabile sull'asse delle classi
            shifted = logits - logits.max(axis=-1, keepdims=True)
            exp = np.exp(shifted)
            probs = exp / exp.sum(axis=-1, keepdims=True)
            # Confidenza = probabilità della classe vincente, per ogni (frame, corda)
            confidences = probs.max(axis=-1)  # (T, 6)
        else:
            confidences = None

        # 4. Post-processing → argmax (tasto predetto)
        risultato = model.post_proc(batch)

    # Estrazione matrice fret per frame → forma (T, 6)
    tab = risultato[tools.KEY_TABLATURE]
    tab_np = tab.cpu().numpy() if isinstance(tab, torch.Tensor) else np.asarray(tab)
    tab_np = np.squeeze(tab_np)
    if tab_np.ndim == 1:
        tab_np = tab_np.reshape(-1, 6)
    elif tab_np.ndim == 3 and tab_np.shape[1] == 6:
        tab_np = tab_np[:, :, 0]
    predictions = tab_np

    # Fallback confidenze se l'estrazione non combacia con le predizioni
    if confidences is None or confidences.shape != predictions.shape:
        logger.warning(
            f"Confidenze non disponibili/compatibili "
            f"({getattr(confidences, 'shape', None)} vs {predictions.shape}): uso 1.0."
        )
        confidences = np.ones_like(predictions, dtype=np.float32)

    logger.info(f"  Predizioni shape: {predictions.shape}  (frame × 6 corde)")
    logger.info(
        f"  Confidenze:       media={confidences.mean():.3f}  "
        f"min={confidences.min():.3f}  max={confidences.max():.3f}"
    )

    return predictions, confidences


# =========================================================================
# Stampa a schermo della tablatura predetta
# =========================================================================
def print_notes_table(notes: list[dict]) -> None:
    """
    Mostra le note trascritte in una tabella leggibile nel terminale.
    """
    if not notes:
        logger.warning("  Nessuna nota trascritta (matrice silenziosa o sotto soglia).")
        return

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"NOTE TRASCRITTE — {len(notes)} note")
    logger.info("=" * 70)

    header = f"  {'#':>3}  {'t(s)':>7}  {'dur(s)':>7}  {'nota':>5}  {'midi':>4}  {'corda':>5}  {'tasto':>5}  {'conf':>5}"
    logger.info(header)
    logger.info("  " + "-" * (len(header) - 2))

    for i, n in enumerate(notes):
        string_idx = n.get("string", -1)
        string_name = (
            config.STRING_NAMES[string_idx]
            if 0 <= string_idx < len(config.STRING_NAMES)
            else "?"
        )
        logger.info(
            f"  {i:>3}  "
            f"{n.get('time', 0):>7.3f}  "
            f"{n.get('duration', 0):>7.3f}  "
            f"{n.get('note_name', '?'):>5}  "
            f"{n.get('pitch', 0):>4}  "
            f"{string_name:>5}  "
            f"{n.get('fret', 0):>5}  "
            f"{n.get('confidence', 0):>5.2f}"
        )


# =========================================================================
# Salvataggio su disco (JSON + CSV)
# =========================================================================
def save_notes(notes: list[dict], npy_path: str, output_dir: str) -> None:
    """
    Salva la sequenza di note in formato .json e .csv.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(npy_path).stem  # es: "00_CMaj_120_cqt"

    json_path = out_path / f"{stem}_notes.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)
    logger.info(f"  Note JSON salvate: {json_path}")

    csv_path = out_path / f"{stem}_notes.csv"
    fieldnames = ["time", "duration", "pitch", "note_name", "string", "fret", "confidence"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(notes)
    logger.info(f"  Note CSV salvate:  {csv_path}")


# =========================================================================
# MAIN — Entry point per esecuzione standalone
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Trascrive una matrice CQT (.npy) in una sequenza di note usando TabCNN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Esempio:\n"
            "  python -m guitar_tutor_pipeline.src.runTabCNN output_cqt/00_CMaj_120_cqt.npy\n"
            "  python -m guitar_tutor_pipeline.src.runTabCNN matrice.npy --output_dir output_notes/\n"
        ),
    )
    parser.add_argument("npy_path", type=str, help="Percorso al file .npy con la matrice CQT (192, T)")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(config.WEIGHTS_PATH),
        help=f"Percorso ai pesi del modello (default: {config.WEIGHTS_PATH})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device su cui eseguire il modello: cpu o cuda (default: cpu)",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=config.HOP_LENGTH,
        help=f"Hop length usato per generare la CQT (per la temporizzazione note, default: {config.HOP_LENGTH})",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=config.SAMPLE_RATE,
        help=f"Sample rate usato per generare la CQT (per la temporizzazione note, default: {config.SAMPLE_RATE})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Cartella dove salvare le note (.json + .csv). Se omesso, stampa solo a schermo.",
    )
    args = parser.parse_args()

    logger.info("🎸 Trascrizione CQT(.npy) → Note con TabCNN")
    logger.info(f"  Input:   {args.npy_path}")
    logger.info(f"  Pesi:    {args.weights}")
    logger.info(f"  Device:  {args.device}")
    logger.info("")

    # ── FASE A: carica la matrice CQT dal .npy ─────────────────────────
    cqt_norm = load_cqt_matrix(args.npy_path)

    # ── FASE B: costruisci il tensore di input ─────────────────────────
    input_tensor = build_input_tensor(cqt_norm, device=args.device)

    # ── Caricamento del modello ────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("Caricamento modello TabCNN")
    logger.info("=" * 70)
    model = load_model(weights_path=args.weights, device=args.device)

    # ── FASE C: inferenza ──────────────────────────────────────────────
    predictions, confidences = run_model(model, input_tensor)

    # ── Decodifica frame-wise → note (riusa il decoder di inference.py) ─
    notes = decode_predictions(
        predictions,
        confidences,
        hop_length=args.hop_length,
        sr=args.sr,
    )

    # ── Output ──────────────────────────────────────────────────────────
    print_notes_table(notes)

    if args.output_dir:
        logger.info("")
        logger.info("=" * 70)
        logger.info("SALVATAGGIO RISULTATI")
        logger.info("=" * 70)
        save_notes(notes, args.npy_path, args.output_dir)

    # ── Riepilogo finale ────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ TRASCRIZIONE COMPLETATA")
    logger.info("=" * 70)
    logger.info(f"  Input:        {args.npy_path}")
    logger.info(f"  Note trovate: {len(notes)}")
    if args.output_dir:
        logger.info(f"  Salvate in:   {args.output_dir}/")


if __name__ == "__main__":
    main()
