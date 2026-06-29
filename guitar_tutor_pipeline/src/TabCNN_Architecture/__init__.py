"""
guitar_tutor_pipeline.src.TabCNN_Architecture — Architettura TabCNN.

Contiene i moduli specifici dell'architettura TabCNN per la trascrizione
automatica della tablatura chitarristica.

Moduli specifici del modello:
    - model: Definizione architettura TabCNN e caricamento pesi
    - inference: Inferenza e trascrizione audio → note
    - audio_processing: Preprocessing audio e estrazione feature CQT

Moduli condivisi (re-esportati da app/):
    - config: Costanti e configurazione centralizzata
    - dataset: Parsing annotazioni JAMS e MIDI
    - alignment: Allineamento DTW e classificazione errori
    - feedback: Generazione feedback pedagogico con LLM
    - pipeline: Orchestrazione end-to-end
"""

# Moduli condivisi — importati da app/
from ..app.config import (
    SAMPLE_RATE,
    HOP_LENGTH,
    N_BINS,
    BINS_PER_OCTAVE,
    WEIGHTS_PATH,
)

from .audio_processing import (
    load_audio,
    compute_cqt,
    prepare_input_tensor,
)

from ..app.dataset import (
    parse_jams,
    parse_midi,
    build_note_sequence,
    midi_to_note_name,
    note_name_to_midi,
    GuitarSetDataset,
)

from .model import (
    TabCNN,
    load_model,
)

from .inference import (
    transcribe_audio,
    decode_predictions,
)

from ..app.alignment import (
    run_alignment,
    compute_dtw_alignment,
    classify_errors,
    build_error_log,
)

from ..app.feedback import (
    generate_feedback,
    build_prompt,
)

from ..app.pipeline import (
    GuitarTutorPipeline,
)

__all__ = [
    # Config
    "SAMPLE_RATE", "HOP_LENGTH", "N_BINS", "BINS_PER_OCTAVE", "WEIGHTS_PATH",
    # Audio
    "load_audio", "compute_cqt", "prepare_input_tensor",
    # Dataset
    "parse_jams", "parse_midi", "build_note_sequence",
    "midi_to_note_name", "note_name_to_midi", "GuitarSetDataset",
    # Model
    "TabCNN", "load_model",
    # Inference
    "transcribe_audio", "decode_predictions",
    # Alignment
    "run_alignment", "compute_dtw_alignment", "classify_errors", "build_error_log",
    # Feedback
    "generate_feedback", "build_prompt",
    # Pipeline
    "GuitarTutorPipeline",
]
