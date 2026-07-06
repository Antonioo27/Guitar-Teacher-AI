# transcribe_real.py  — uso: python transcribe_real.py data/real_recordings/scala_do.wav
import sys
import argparse
from pathlib import Path

# Aggiunge la root del progetto al path se lo script viene lanciato direttamente
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import pretty_midi
import soundfile as sf
import matplotlib
matplotlib.use("Agg")               # niente display, solo salvataggio file
import matplotlib.pyplot as plt

from guitar_tutor_pipeline.src.CRNN_Model.config import config
from guitar_tutor_pipeline.src.CRNN_Model.model import (
    Regress_onset_offset_frame_velocity_CRNN,
    load_finetuned_checkpoint,
)
from guitar_tutor_pipeline.src.CRNN_Model.inference import transcribe_real

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_piano_roll(notes, out_png):
    fig, ax = plt.subplots(figsize=(14, 5))
    for onset, offset, pitch, _ in notes:
        ax.plot([onset, offset], [pitch, pitch], lw=4, alpha=0.7)
    ax.set_xlabel("Tempo (s)"); ax.set_ylabel("Pitch MIDI")
    ax.set_title(f"{len(notes)} note"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=100); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=Path, help="file .wav da trascrivere")
    ap.add_argument("--thresh", type=float, default=0.05, help="soglia onset (default 0.05, quella tarata)")
    ap.add_argument("--program", type=int, default=24, help="24=nylon 25=steel 26=jazz electric")
    args = ap.parse_args()

    if not args.wav.exists():
        sys.exit(f"File non trovato: {args.wav}")

    out_dir = args.wav.parent / "output"
    out_dir.mkdir(exist_ok=True)
    stem = args.wav.stem
    midi_path    = out_dir / f"{stem}.mid"
    roll_path    = out_dir / f"{stem}_pianoroll.png"
    preview_path = out_dir / f"{stem}_preview.wav"

    print(f"Device: {device}  |  soglia: {args.thresh}")
    print("Caricamento modello...")
    model = Regress_onset_offset_frame_velocity_CRNN(
        config.FRAMES_PER_SECOND, config.CLASSES_NUM).to(device)
    model = load_finetuned_checkpoint(model, config.BEST_MODEL_PATH, device=device)

    # 1. trascrizione -> .mid
    notes = transcribe_real(model, args.wav, midi_path, device,
                            onset_thresh=args.thresh)

    if not notes:
        print("Nessuna nota rilevata. Controlla livello audio / soglia.")
        return

    # 2. piano-roll per ispezione visiva
    plot_piano_roll(notes, roll_path)
    print(f"Piano-roll -> {roll_path}")

    # 3. preview audio (sintesi a seni, per controllo a orecchio)
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    sf.write(str(preview_path), pm.synthesize(fs=16000), 16000)
    print(f"Preview -> {preview_path}")

    # 4. riepilogo note (utile per le scale, dove conosci le note attese)
    print(f"\n{len(notes)} note:")
    for onset, offset, pitch, vel in sorted(notes, key=lambda n: n[0]):
        name = pretty_midi.note_number_to_name(pitch)
        print(f"  t={onset:5.2f}s  {name:<4} (pitch {pitch})  dur={offset-onset:.2f}s")
        if len(notes) > 30:
            print(f"  ... e altre {len(notes)-30}")


if __name__ == "__main__":
    main()