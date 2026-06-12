"""
config.py — Costanti e configurazione centralizzata per la pipeline AI Guitar Tutor.

Tutti i parametri globali (audio, modello, DTW, LLM) sono definiti qui
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Guitar-Teacher-AI/
PIPELINE_ROOT = PROJECT_ROOT / "guitar_tutor_pipeline"
DATA_DIR = PIPELINE_ROOT / "data"
GUITARSET_DIR = DATA_DIR / "GuitarSet"
WEIGHTS_DIR = DATA_DIR

# =============================================================================
# Modulo 1 — Parametri Audio & CQT
# =============================================================================
SAMPLE_RATE = 44100           # Frequenza di campionamento standard (Hz)
HOP_LENGTH = 512              # Hop length per la CQT (campioni)
N_BINS = 192                  # Numero totale di bin frequenziali nella CQT
BINS_PER_OCTAVE = 24          # Risoluzione frequenziale (24 = quarti di tono)
FMIN = 32.70319566257483      # Frequenza minima per la CQT (C1)
AUDIO_DURATION = None         # Durata massima audio in secondi (None = intero file)

# =============================================================================
# Modulo 2 — Parametri del Modello TabCNN
# =============================================================================
NUM_STRINGS = 6               # Numero di corde della chitarra
NUM_FRETS = 20                # Fret 0..19 (0 = corda a vuoto)
NUM_CLASSES = NUM_FRETS + 1   # 21 classi per corda (0-19 + "non suonata")
# Il modello predice per ogni frame e per ogni corda quale tasto è premuto

ONSET_THRESHOLD = 0.5
WEIGHTS_FILENAME = "our weights/GuitarSet_best_validation.pt"  # Nome del file dei pesi
WEIGHTS_PATH = WEIGHTS_DIR / WEIGHTS_FILENAME

# Dimensioni di input attese dal modello TabCNN
CONTEXT_FRAMES = 9            # Frame di contesto temporale (finestra locale)

# =============================================================================
# Modulo 3 — Parametri DTW / Alignment
# =============================================================================
TIME_TOLERANCE = 0.1          # Tolleranza temporale in secondi per "a tempo"
PITCH_WEIGHT = 1.0            # Peso della distanza di pitch nel DTW
TIME_WEIGHT = 0.5             # Peso della distanza temporale nel DTW

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
