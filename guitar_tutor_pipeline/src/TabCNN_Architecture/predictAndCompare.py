"""
predictAndCompare.py — Orchestratore: da matrice CQT (.npy) + spartito (.mid)
a predizione + confronto visivo, in un solo comando.

Incatena i due step della pipeline a valle della CQT:

    .npy ─(TabCNN)→ note predette ─(DTW vs MIDI)→ report + immagini

Riusa i blocchi già esistenti:
    - runTabCNN:      load_cqt_matrix, build_input_tensor, run_model,
                      detect_onset_times, resolve_ghost_notes, save_notes
    - inference:      decode_predictions
    - model:          load_model
    - compareWithMidi: run_comparison (DTW + piano roll + overlay)

Output (in --output_dir):
    <stem>_notes.json / .csv     → note predette
    <stem>_vs_midi.png           → piano roll a due corsie (collegamenti DTW)
    <stem>_overlay.png           → sovrapposizione sullo stesso asse (sfasamento)

NOTA sui tempi: di default usa lo stesso SAMPLE_RATE/HOP_LENGTH con cui
convertWawToCQT.py genera i .npy, così i tempi delle note predette sono
coerenti con quelli del MIDI (e l'overlay è leggibile).

Uso:
    python -m guitar_tutor_pipeline.src.predictAndCompare \\
        output_cqt/velocity127/00_CMaj_120_Velocity127_cqt.npy \\
        SetTestHomeMade/00_CMaj_120_Velocity127.mid \\
        --output_dir output_notes/velocity127
"""

import sys
import logging
import argparse
from pathlib import Path

from ..app import config
from .model import load_model
from .inference import decode_predictions
from ..app.dataset import parse_midi
from .convertWawToCQT import SAMPLE_RATE as CQT_SR, HOP_LENGTH as CQT_HOP
from .runTabCNN import (
    load_cqt_matrix,
    build_input_tensor,
    run_model,
    detect_onset_times,
    resolve_ghost_notes,
    filter_short_notes,
    save_notes,
)
from .compareWithMidi import run_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Da .npy + .mid: predizione TabCNN + confronto DTW + immagini, in un comando.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Esempio:\n"
            "  python -m guitar_tutor_pipeline.src.predictAndCompare \\\n"
            "    output_cqt/velocity127/00_CMaj_120_Velocity127_cqt.npy \\\n"
            "    SetTestHomeMade/00_CMaj_120_Velocity127.mid \\\n"
            "    --output_dir output_notes/velocity127\n"
        ),
    )
    parser.add_argument("npy_path", type=str, help="Matrice CQT .npy (output di convertWawToCQT)")
    parser.add_argument("midi_path", type=str, help="Spartito .mid/.midi di riferimento")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_notes",
        help="Cartella dove salvare note e immagini (default: output_notes/)",
    )
    parser.add_argument("--weights", type=str, default=str(config.WEIGHTS_PATH),
                        help=f"Pesi del modello (default: {config.WEIGHTS_PATH})")
    parser.add_argument("--device", type=str, default="cpu", help="cpu o cuda (default: cpu)")
    parser.add_argument("--sr", type=int, default=CQT_SR,
                        help=f"Sample rate usato per generare il .npy (default: {CQT_SR})")
    parser.add_argument("--hop_length", type=int, default=CQT_HOP,
                        help=f"Hop length usato per generare il .npy (default: {CQT_HOP})")
    parser.add_argument("--gap-fill", type=float, default=0.06,
                        help="Gap fill (s) nel decoder: unisce solo il chattering, non i ri-pizzicati (default: 0.06).")
    parser.add_argument("--min-duration", type=float, default=0.12,
                        help="Durata minima (s): note più corte vengono scartate come artefatti (default: 0.12).")
    parser.add_argument("--no-ghost-fix", action="store_true",
                        help="Disattiva la risoluzione delle ghost note.")
    parser.add_argument("--ghost-gap", type=float, default=0.06,
                        help="Gap di anti-chattering (s) per fondere frammenti dello stesso pitch (default: 0.06).")
    parser.add_argument("--no-onsets", action="store_true",
                        help="Non usare gli onset della CQT nella risoluzione ghost.")
    parser.add_argument("--onset-delta", type=float, default=0.07,
                        help="Soglia di prominenza per il rilevamento onset (default: 0.07).")
    args = parser.parse_args()

    if not Path(args.npy_path).exists():
        logger.error(f"File .npy non trovato: {args.npy_path}")
        sys.exit(1)
    if not Path(args.midi_path).exists():
        logger.error(f"File MIDI non trovato: {args.midi_path}")
        sys.exit(1)

    logger.info("🎸→🎯 Predizione + confronto (npy + mid)")
    logger.info(f"  CQT:      {args.npy_path}")
    logger.info(f"  Spartito: {args.midi_path}")
    logger.info(f"  sr={args.sr}  hop={args.hop_length}  device={args.device}")
    logger.info("")

    # ── STEP 1: predizione (.npy → note) ────────────────────────────────
    cqt_norm = load_cqt_matrix(args.npy_path)
    input_tensor = build_input_tensor(cqt_norm, device=args.device)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Caricamento modello TabCNN")
    logger.info("=" * 70)
    model = load_model(weights_path=args.weights, device=args.device)

    predictions, confidences = run_model(model, input_tensor)
    notes = decode_predictions(
        predictions, confidences, hop_length=args.hop_length, sr=args.sr,
        gap_fill_s=args.gap_fill,
    )

    if not args.no_ghost_fix:
        onset_times = None
        if not args.no_onsets:
            onset_times = detect_onset_times(
                cqt_norm, hop_length=args.hop_length, sr=args.sr, delta=args.onset_delta
            )
        before = len(notes)
        notes = resolve_ghost_notes(notes, max_gap_s=args.ghost_gap, onset_times=onset_times)
        logger.info(
            f"  Ghost-fix: {before} → {len(notes)} note (fuse {before - len(notes)})"
        )

    # Filtro durata minima: rimuove i micro-frammenti residui
    notes = filter_short_notes(notes, min_duration_s=args.min_duration)

    logger.info(f"  Note predette: {len(notes)}")

    # Salva note (.json + .csv) — stesso stem del .npy
    logger.info("")
    logger.info("=" * 70)
    logger.info("SALVATAGGIO NOTE")
    logger.info("=" * 70)
    save_notes(notes, args.npy_path, args.output_dir)

    # ── STEP 2: confronto col MIDI (DTW + immagini) ─────────────────────
    reference = parse_midi(args.midi_path)
    logger.info(f"  Note spartito: {len(reference)}")

    run_comparison(
        notes, reference, args.midi_path,
        output_dir=args.output_dir,
        stem=Path(args.npy_path).stem,
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ FATTO")
    logger.info("=" * 70)
    logger.info(f"  Tutto salvato in: {args.output_dir}/")


if __name__ == "__main__":
    main()
