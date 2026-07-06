"""
Scarica il dataset GAPS e crea la cartella del test set in:
    <PIPELINE_ROOT>/data/gaps_test/audio/
    <PIPELINE_ROOT>/data/gaps_test/midi/

Lo split usa random.seed(42) + shuffle → ultimi 10% = test (stessa proporzione
del notebook; il seed fisso garantisce che il test set sia sempre identico).

Uso:
    python -m guitar_tutor_pipeline.src.CRNN_Model.prepare_test_set
    oppure direttamente:
    python guitar_tutor_pipeline/src/CRNN_Model/prepare_test_set.py
"""

import random
import shutil
import sys
import urllib.request
from pathlib import Path

try:
    from .config import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from guitar_tutor_pipeline.src.CRNN_Model.config import config

SEED = 42
DATA_ROOT  = config.PIPELINE_ROOT / "data"
SPLIT_FILE = DATA_ROOT / "split.json"      # unica fonte di verità per lo split
VAL_DIR    = DATA_ROOT / "gaps_val"
TEST_DIR   = DATA_ROOT / "gaps_test"


def download_gaps():
    from huggingface_hub import snapshot_download

    audio_dir = config.DATA_DIR / "audio"
    if audio_dir.exists() and any(audio_dir.glob("*.wav")):
        print(f"GAPS già presente in {config.DATA_DIR}")
        return

    print("Scaricamento GAPS dataset da HuggingFace...")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="xavriley/GAPS",
        repo_type="dataset",
        local_dir=str(config.DATA_DIR),
        allow_patterns=["audio/*.wav", "midi/*.mid"],
    )
    print("Download completato.")


def download_maestro_checkpoint():
    url = (
        "https://zenodo.org/record/4034264/files/"
        "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
    )
    ckpt = config.MAESTRO_CHECKPOINT
    if ckpt.exists() and ckpt.stat().st_size > 1_000_000:
        print(f"Checkpoint MAESTRO già presente: {ckpt}")
        return

    print("Download pesi MAESTRO da Zenodo...")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(ckpt))
    print(f"Checkpoint salvato ({ckpt.stat().st_size / 1e6:.1f} MB): {ckpt}")


def build_all_pairs():
    audio_dir = config.DATA_DIR / "audio"
    midi_dir  = config.DATA_DIR / "midi"
    pairs = [
        (audio_dir / f.name.replace(".mid", ".wav"), f)
        for f in sorted(midi_dir.glob("*.mid"))
    ]
    return [(a, m) for a, m in pairs if a.exists()]


def build_split(all_pairs):
    """Carica split.json se esiste (es. recuperato da Colab), altrimenti lo crea.
    Lo split è memorizzato per NOME MIDI: portabile tra macchine/percorsi."""
    if SPLIT_FILE.exists():
        print(f"Split esistente caricato: {SPLIT_FILE}")
        return json.loads(SPLIT_FILE.read_text())

    pairs = sorted(all_pairs, key=lambda p: p[1].name)   # ordine canonico esplicito
    rng = random.Random(SEED)                            # RNG isolato, non tocca lo stato globale
    rng.shuffle(pairs)
    n = len(pairs)
    n_tr, n_va = int(n * 0.8), int(n * 0.1)
    split = {
        "train": [m.name for _, m in pairs[:n_tr]],
        "val":   [m.name for _, m in pairs[n_tr:n_tr + n_va]],
        "test":  [m.name for _, m in pairs[n_tr + n_va:]],
    }
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps(split, indent=2))
    print(f"Nuovo split creato e salvato: {SPLIT_FILE}")
    print("  -> USA QUESTO STESSO split.json anche per il training, altrimenti contamini il test.")
    return split


def materialize(subset_name, midi_names, by_name, dst_dir):
    if list(dst_dir.glob("audio/*.wav")):
        print(f"  {subset_name}: cartella già presente, salto.")
        return
    (dst_dir / "audio").mkdir(parents=True, exist_ok=True)
    (dst_dir / "midi").mkdir(parents=True, exist_ok=True)
    for name in midi_names:
        wav, mid = by_name[name]
        shutil.copy2(wav, dst_dir / "audio" / wav.name)
        shutil.copy2(mid, dst_dir / "midi"  / mid.name)
    print(f"  {subset_name}: {len(midi_names)} coppie -> {dst_dir}")


def main():
    download_gaps()
    download_maestro_checkpoint()

    all_pairs = build_all_pairs()
    by_name = {m.name: (a, m) for a, m in all_pairs}
    print(f"Coppie totali GAPS: {len(all_pairs)}")

    split = build_split(all_pairs)
    print(f"Split -> train: {len(split['train'])}  val: {len(split['val'])}  test: {len(split['test'])}")

    # sanity check: nessun file condiviso tra i tre insiemi
    s_tr, s_va, s_te = set(split["train"]), set(split["val"]), set(split["test"])
    assert s_tr.isdisjoint(s_va) and s_tr.isdisjoint(s_te) and s_va.isdisjoint(s_te), \
        "ERRORE: overlap tra train/val/test!"

    # in locale materializzo solo val + test (il train non serve per inferenza/eval)
    materialize("val",  split["val"],  by_name, VAL_DIR)
    materialize("test", split["test"], by_name, TEST_DIR)


if __name__ == "__main__":
    main()
