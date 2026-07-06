"""
inspectMidi.py — Script standalone per ispezionare un file MIDI di annotazione.

Serve a vedere "cosa c'è dentro" un file .mid (es. la traccia esportata da
Logic) e a esportarne le note in CSV, con le STESSE colonne usate da
runTabCNN.py. In questo modo puoi mettere fianco a fianco:

    annotazione (MIDI)  vs  predizione del modello (runTabCNN)

e confrontarle a occhio.

Output salvato (con --output_dir):
    - <nome>_midi.csv   → note dell'annotazione in formato tabellare

Uso:
    python -m guitar_tutor_pipeline.src.TabCNN_Architecture.inspectMidi SetTestHomeMade/00_CMaj_120.mid
    python -m guitar_tutor_pipeline.src.TabCNN_Architecture.inspectMidi file.mid --output_dir output_notes/
"""

import sys
import csv
import logging
import argparse
from pathlib import Path

from ..app.dataset import parse_midi

# =========================================================================
# Configurazione del logger — coerente con gli altri script della pipeline
# =========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =========================================================================
# Stampa a schermo delle note del MIDI
# =========================================================================
def print_midi_table(notes: list[dict]) -> None:
    """
    Mostra le note dell'annotazione MIDI in una tabella leggibile.

    Colonne: indice, tempo di inizio, durata, nome nota, pitch MIDI, velocity.
    """
    if not notes:
        logger.warning("  Nessuna nota trovata nel file MIDI.")
        return

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"NOTE MIDI (annotazione) — {len(notes)} note")
    logger.info("=" * 70)

    header = f"  {'#':>3}  {'t(s)':>7}  {'dur(s)':>7}  {'nota':>5}  {'midi':>4}  {'vel':>4}"
    logger.info(header)
    logger.info("  " + "-" * (len(header) - 2))

    for i, n in enumerate(notes):
        logger.info(
            f"  {i:>3}  "
            f"{n.get('time', 0):>7.3f}  "
            f"{n.get('duration', 0):>7.3f}  "
            f"{n.get('note_name', '?'):>5}  "
            f"{n.get('midi_pitch', 0):>4}  "
            f"{n.get('velocity', 0):>4}"
        )


# =========================================================================
# Salvataggio su disco (CSV)
# =========================================================================
def save_midi_csv(notes: list[dict], midi_path: str, output_dir: str) -> None:
    """
    Salva le note dell'annotazione MIDI in formato .csv.

    Le colonne combaciano (per quanto possibile) con quelle del CSV prodotto
    da runTabCNN.py, per facilitare il confronto.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(midi_path).stem

    csv_path = out_path / f"{stem}_midi.csv"
    fieldnames = ["time", "duration", "midi_pitch", "note_name", "velocity"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(notes)
    logger.info(f"  Note MIDI CSV salvate: {csv_path}")


# =========================================================================
# MAIN — Entry point per esecuzione standalone
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Ispeziona un file MIDI di annotazione ed esporta le note in CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Esempio:\n"
            "  python -m guitar_tutor_pipeline.src.inspectMidi SetTestHomeMade/00_CMaj_120.mid\n"
            "  python -m guitar_tutor_pipeline.src.inspectMidi file.mid --output_dir output_notes/\n"
        ),
    )
    parser.add_argument("midi_path", type=str, help="Percorso al file .mid/.midi da ispezionare")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Cartella dove salvare il CSV. Se omesso, stampa solo a schermo.",
    )
    args = parser.parse_args()

    if not Path(args.midi_path).exists():
        logger.error(f"File MIDI non trovato: {args.midi_path}")
        sys.exit(1)

    logger.info("🎼 Ispezione file MIDI (annotazione)")
    logger.info(f"  Input: {args.midi_path}")

    # parse_midi() legge il MIDI con pretty_midi e restituisce la lista di note
    notes = parse_midi(args.midi_path)

    # Statistiche di riepilogo
    if notes:
        end_time = max(n["time"] + n["duration"] for n in notes)
        pitches = sorted({n["note_name"] for n in notes})
        logger.info(f"  Note totali: {len(notes)}")
        logger.info(f"  Durata:      {end_time:.3f} s")
        logger.info(f"  Pitch usati: {', '.join(pitches)}")

    print_midi_table(notes)

    if args.output_dir:
        logger.info("")
        logger.info("=" * 70)
        logger.info("SALVATAGGIO CSV")
        logger.info("=" * 70)
        save_midi_csv(notes, args.midi_path, args.output_dir)

    logger.info("")
    logger.info("✅ Ispezione completata")


if __name__ == "__main__":
    main()
