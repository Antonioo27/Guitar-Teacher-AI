"""
guitar_tutor_pipeline.src — AI Guitar Tutor Pipeline.

Pipeline ibrida (Deep Learning + DTW + LLM) per la valutazione
dell'esecuzione chitarristica.

Moduli:
    - audio_processing: Preprocessing audio e estrazione feature CQT
    - dataset: Parsing annotazioni JAMS e MIDI
    - model: Definizione architettura TabCNN e caricamento pesi
    - inference: Inferenza e trascrizione audio → note
    - alignment: Allineamento DTW e classificazione errori
    - feedback: Generazione feedback pedagogico con LLM
    - pipeline: Orchestrazione end-to-end
    - config: Costanti e configurazione
"""

from .config import (
    SAMPLE_RATE,
    HOP_LENGTH,
    N_BINS,
    BINS_PER_OCTAVE,
    WEIGHTS_PATH,
)

from .audio_processing import (
    load_audio,
    compute_cqt,
    normalize_cqt,
    prepare_input_tensor,
)

from .dataset import (
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
    predict_tablature,
    decode_predictions,
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
    # Audio
    "load_audio", "compute_cqt", "normalize_cqt", "prepare_input_tensor",
    # Dataset
    "parse_jams", "parse_midi", "build_note_sequence",
    "midi_to_note_name", "note_name_to_midi", "GuitarSetDataset",
    # Model
    "TabCNN", "load_model",
    # Inference
    "transcribe_audio", "predict_tablature", "decode_predictions",
    # Alignment
    "run_alignment", "compute_dtw_alignment", "classify_errors", "build_error_log",
    # Feedback
    "generate_feedback", "build_prompt",
    # Pipeline
    "GuitarTutorPipeline",
]
