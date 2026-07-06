# build_split.py  — esegui UNA volta, poi non toccarlo più
import re, json, random, sys
from pathlib import Path

try:
    from .config import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from guitar_tutor_pipeline.src.CRNN_Model.config import config

SEED = 42
DATA_ROOT  = config.PIPELINE_ROOT / "data"
RAW_FILE   = DATA_ROOT / "colab_eval_output.txt"   # ci incolli l'output di Colab
SPLIT_FILE = DATA_ROOT / "split.json"

raw = RAW_FILE.read_text()
test_names = sorted({m.replace(".wav", ".mid") for m in re.findall(r"(\S+\.wav)", raw)})
assert len(test_names) == 41, f"Attesi 41 file, trovati {len(test_names)} — controlla l'output incollato"

# sub-split deterministico dei 41: dev (taratura soglia) + test (reporting)
shuffled = list(test_names)
random.Random(SEED).shuffle(shuffled)
k = len(shuffled) // 2                       # 20
split = {
    "recovered_colab_test": test_names,      # i 41 originali, per tracciabilità
    "dev":  sorted(shuffled[:k]),            # 20 file -> SOLO per scegliere la soglia
    "test": sorted(shuffled[k:]),            # 21 file -> numero finale, una volta sola
}

# i tre insiemi che useremo devono essere disgiunti
assert set(split["dev"]).isdisjoint(split["test"])

SPLIT_FILE.write_text(json.dumps(split, indent=2))
print(f"split.json scritto: dev={len(split['dev'])}  test={len(split['test'])}")
print("Questi 41 NON sono mai stati visti in training (recuperati dal log di Colab).")