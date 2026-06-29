"""
guitar_tutor_pipeline.src.app — Moduli condivisi dell'AI Guitar Tutor.

Contiene i moduli trasversali della pipeline, indipendenti
dall'architettura del modello di trascrizione utilizzato:
    - config: Costanti e configurazione centralizzata
    - alignment: Allineamento DTW e classificazione errori
    - feedback: Generazione feedback pedagogico con LLM
    - dataset: Parsing annotazioni JAMS e MIDI
    - pipeline: Orchestrazione end-to-end
    - model_registry: Registry per il caricamento dinamico dei modelli
"""

from .config import (
    SAMPLE_RATE,
    HOP_LENGTH,
    N_BINS,
    BINS_PER_OCTAVE,
    WEIGHTS_PATH,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
)

from .dataset import (
    parse_jams,
    parse_midi,
    build_note_sequence,
    midi_to_note_name,
    note_name_to_midi,
    GuitarSetDataset,
)

from .alignment import (
    run_alignment,
    compute_dtw_alignment,
    classify_errors,
    build_error_log,
)

from .feedback import (
    generate_feedback,
    build_prompt,
)

from .pipeline import (
    GuitarTutorPipeline,
)

__all__ = [
    # Config
    "SAMPLE_RATE", "HOP_LENGTH", "N_BINS", "BINS_PER_OCTAVE", "WEIGHTS_PATH",
    "AVAILABLE_MODELS", "DEFAULT_MODEL",
    # Dataset
    "parse_jams", "parse_midi", "build_note_sequence",
    "midi_to_note_name", "note_name_to_midi", "GuitarSetDataset",
    # Alignment
    "run_alignment", "compute_dtw_alignment", "classify_errors", "build_error_log",
    # Feedback
    "generate_feedback", "build_prompt",
    # Pipeline
    "GuitarTutorPipeline",
]
