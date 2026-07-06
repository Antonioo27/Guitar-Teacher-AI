# split.py — solo helper, niente codice al top-level
import json
import sys
from pathlib import Path

try:
    from .config import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from guitar_tutor_pipeline.src.CRNN_Model.config import config


def load_pairs(subset):  # "dev" oppure "test"
    split = json.loads((config.PIPELINE_ROOT / "data" / "split.json").read_text())
    audio_dir, midi_dir = config.DATA_DIR / "audio", config.DATA_DIR / "midi"
    pairs = []
    for mid_name in split[subset]:
        mid = midi_dir / mid_name
        wav = audio_dir / mid_name.replace(".mid", ".wav")
        if wav.exists() and mid.exists():
            pairs.append((wav, mid))
        else:
            print(f"  ATTENZIONE: manca {mid_name}, lo salto")
    return pairs