# evaluate_synthetic.py
import numpy as np
import torch

from guitar_tutor_pipeline.src.CRNN_Model.config import config
from guitar_tutor_pipeline.src.CRNN_Model.model import (
    Regress_onset_offset_frame_velocity_CRNN,
    load_finetuned_checkpoint,
)

from guitar_tutor_pipeline.src.CRNN_Model.inference import prf_real

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REC_DIR  = config.PIPELINE_ROOT / "data" / "syntetic_recordings"
REF_DIR  = REC_DIR / "midi"          # <-- ADATTA: dove tieni i .mid di RIFERIMENTO esportati da Logic
ONSET_TH = 0.05


def find_ref(wav):
    """Cerca il .mid di riferimento con lo stesso stem del .wav."""
    ref = REF_DIR / (wav.stem + ".mid")
    return ref if ref.exists() else None


if __name__ == "__main__":
    print(f"Device: {device}  |  soglia: {ONSET_TH}")
    model = Regress_onset_offset_frame_velocity_CRNN(
        config.FRAMES_PER_SECOND, config.CLASSES_NUM).to(device)
    model = load_finetuned_checkpoint(model, config.BEST_MODEL_PATH, device=device)

    wavs = sorted(REC_DIR.glob("*.wav"))
    print(f"Trovati {len(wavs)} file .wav\n")

    rows = []
    print(f"{'file':<28} {'P':>6} {'R':>6} {'F1':>6}")
    for wav in wavs:
        ref = find_ref(wav)
        if ref is None:
            print(f"{wav.name:<28}  --- nessun riferimento in {REF_DIR}")
            continue
        # GUARDIA: il riferimento non deve mai essere il MIDI predetto (sta in output/)
        assert "output" not in ref.parts, \
            f"ERRORE: {ref} sembra il MIDI PREDETTO, non il riferimento!"

        res = prf_real(model, wav, ref, device, onset_thresh=ONSET_TH)
        if res is None:
            print(f"{wav.name:<28}  --- nessuna nota (ref o predetto vuoto)")
            continue
        P, R, F1 = res
        rows.append((wav.name, P, R, F1))
        print(f"{wav.name:<28} {P*100:6.1f} {R*100:6.1f} {F1*100:6.1f}")

    if rows:
        P = np.array([r[1] for r in rows])
        R = np.array([r[2] for r in rows])
        F = np.array([r[3] for r in rows])
        print("=" * 50)
        print(f"MEDIA ({len(rows)} file)  "
              f"P={P.mean()*100:.1f}±{P.std()*100:.1f}  "
              f"R={R.mean()*100:.1f}±{R.std()*100:.1f}  "
              f"F1={F.mean()*100:.1f}±{F.std()*100:.1f}")