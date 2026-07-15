"""
config.py — Costanti e configurazione centralizzata per la pipeline AI Guitar Tutor.

Tutti i parametri globali (audio, modello, alignment, LLM) sono definiti qui
per garantire coerenza tra i moduli.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Carica variabili d'ambiente da file .env (se presente)
load_dotenv()

# =============================================================================
# Percorsi del progetto
# =============================================================================
# app/ → src/ → guitar_tutor_pipeline/ → Guitar-Teacher-AI/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PIPELINE_ROOT = PROJECT_ROOT / "guitar_tutor_pipeline"
DATA_DIR = PIPELINE_ROOT / "data"
GUITARSET_DIR = DATA_DIR / "GuitarSet"
WEIGHTS_DIR = PIPELINE_ROOT / "weights"
WEIGHTS_DIR_SYNTHTAB = PIPELINE_ROOT / "data/SynthTabWeights"

# =============================================================================
# Selezione Modello — Configurazione Multi-Modello
# =============================================================================
AVAILABLE_MODELS = ["TabCNN", "CRNN"]
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "TabCNN")

# =============================================================================
# Modulo 1 — Parametri Audio & CQT (usati da TabCNN)
# =============================================================================
SAMPLE_RATE = 22050           # Frequenza di campionamento standard (Hz)
HOP_LENGTH = 512              # Hop length per la CQT (campioni)
N_BINS = 192                  # Numero totale di bin frequenziali nella CQT
BINS_PER_OCTAVE = 24          # Risoluzione frequenziale (24 = quarti di tono)
FMIN = 32.70319566257483      # Frequenza minima: C1 (usata nel paper TabCNN)
AUDIO_DURATION = None         # Durata massima audio in secondi (None = intero file)

# =============================================================================
# Modulo 2a — Parametri del Modello TabCNN
# =============================================================================
NUM_STRINGS = 6               # Numero di corde della chitarra
NUM_FRETS = 20                # Fret 0..19 (0 = corda a vuoto)
NUM_CLASSES = NUM_FRETS + 1   # 21 classi per corda (0-19 + "non suonata")
# Il modello predice per ogni frame e per ogni corda quale tasto è premuto

ONSET_THRESHOLD = 0.5         # Soglia di confidenza per la detection delle note
WEIGHTS_FILENAME = "GuitarSet.pt"  # Nome del file dei pesi TabCNN
WEIGHTS_PATH = WEIGHTS_DIR / WEIGHTS_FILENAME

# Dimensioni di input attese dal modello TabCNN
CONTEXT_FRAMES = 9            # Frame di contesto temporale (finestra locale)

# =============================================================================
# Modulo 2b — Parametri del
#  Modello CRNN
# =============================================================================
CRNN_WEIGHTS_FILENAME = "maestro_model_finetune.pth"  # Nome del file dei pesi CRNN
CRNN_WEIGHTS_PATH = WEIGHTS_DIR / CRNN_WEIGHTS_FILENAME
CRNN_BEGIN_NOTE = 21          # MIDI pitch della nota più bassa (A0)
CRNN_CLASSES_NUM = 88         # Numero di classi (tasti pianoforte)

# =============================================================================
# Modulo 3 — Parametri Alignment
# =============================================================================
TIME_TOLERANCE = 0.1          # Tolleranza temporale in secondi per "a tempo"

# =============================================================================
# Modulo 4 — Parametri LLM / Feedback
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# =============================================================================
# Mapping MIDI → Nomi Note
# =============================================================================
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Accordatura standard della chitarra (MIDI pitch delle corde a vuoto)
GUITAR_TUNING = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4
STRING_NAMES = ["E2", "A2", "D3", "G3", "B3", "E4"]
