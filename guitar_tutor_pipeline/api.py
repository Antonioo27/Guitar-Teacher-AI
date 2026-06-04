"""
api.py — REST API per il frontend Vue dell'AI Guitar Tutor.

Espone la pipeline come servizio web con FastAPI:
- Upload file audio (.wav)
- Upload spartito di riferimento (.mid/.jams)
- Esecuzione della pipeline
- Restituzione dei risultati (errori + feedback LLM)

Avvio:
    source venv/bin/activate
    uvicorn guitar_tutor_pipeline.api:app --reload --port 8000
"""

import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .src import config
from .src.model import load_model, TabCNN
from .src.inference import transcribe_audio
from .src.alignment import run_alignment
from .src.feedback import generate_feedback
from .src.dataset import parse_midi, parse_jams, build_note_sequence

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# =========================================================================
# FastAPI App
# =========================================================================
app = FastAPI(
    title="AI Guitar Tutor",
    description="Trascrizione neurale e valutazione dell'esecuzione chitarristica",
    version="1.0.0",
)

# CORS — permetti il frontend Vue (dev su porta 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================================
# Caricamento del modello (una volta sola all'avvio)
# =========================================================================
_model: Optional[TabCNN] = None


def get_model() -> TabCNN:
    """Lazy-load del modello TabCNN."""
    global _model
    if _model is None:
        try:
            _model = load_model(config.WEIGHTS_PATH)
            logger.info("Modello TabCNN caricato con successo.")
        except FileNotFoundError as e:
            logger.error(f"Pesi del modello non trovati: {e}")
            raise
    return _model


# =========================================================================
# Directory per upload temporanei
# =========================================================================
UPLOAD_DIR = config.PIPELINE_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def _save_upload(upload: UploadFile, suffix: str) -> Path:
    """Salva un file uploadato in una directory temporanea."""
    dest = UPLOAD_DIR / f"{upload.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


# =========================================================================
# Endpoints
# =========================================================================

@app.get("/api/health")
async def health_check():
    """Verifica che il server sia attivo."""
    model_loaded = _model is not None
    weights_exist = config.WEIGHTS_PATH.exists()
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "weights_available": weights_exist,
        "weights_path": str(config.WEIGHTS_PATH),
    }


@app.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(..., description="File audio .wav dello studente"),
):
    """
    Moduli 1+2: Trascrivi un file audio in una sequenza di note.
    Restituisce le note predette senza confronto con lo spartito.
    """
    if not audio.filename.lower().endswith((".wav", ".mp3", ".flac")):
        raise HTTPException(400, "Formato audio non supportato. Usa .wav, .mp3 o .flac")

    audio_path = _save_upload(audio, ".wav")

    try:
        model = get_model()
        notes = transcribe_audio(str(audio_path), model)
        return {"notes": notes, "total_notes": len(notes)}
    except FileNotFoundError:
        raise HTTPException(
            503,
            "Modello non disponibile. Assicurati che i pesi siano in "
            f"{config.WEIGHTS_DIR}/{config.WEIGHTS_FILENAME}"
        )
    except Exception as e:
        logger.error(f"Errore nella trascrizione: {e}", exc_info=True)
        raise HTTPException(500, f"Errore nella trascrizione: {str(e)}")
    finally:
        audio_path.unlink(missing_ok=True)


@app.post("/api/analyze")
async def analyze(
    audio: UploadFile = File(..., description="File audio .wav dello studente"),
    reference: UploadFile = File(..., description="Spartito di riferimento (.mid, .jams)"),
    context: str = Form(default="", description="Descrizione dell'esercizio"),
    generate_feedback_flag: bool = Form(
        default=True, alias="generate_feedback",
        description="Se generare il feedback LLM",
    ),
    time_tolerance: float = Form(
        default=config.TIME_TOLERANCE,
        description="Tolleranza temporale in secondi",
    ),
):
    """
    Pipeline completa: trascrizione + allineamento DTW + feedback LLM.

    Richiede:
    - File audio dell'esecuzione dello studente
    - File di riferimento dello spartito (MIDI o JAMS)
    - (Opzionale) Contesto dell'esercizio per il feedback LLM
    """
    # Validazione formato audio
    if not audio.filename.lower().endswith((".wav", ".mp3", ".flac")):
        raise HTTPException(400, "Formato audio non supportato. Usa .wav, .mp3 o .flac")

    # Validazione formato riferimento
    if not reference.filename.lower().endswith((".mid", ".midi", ".jams")):
        raise HTTPException(400, "Formato spartito non supportato. Usa .mid, .midi o .jams")

    audio_path = _save_upload(audio, ".wav")
    ref_path = _save_upload(reference, Path(reference.filename).suffix)

    try:
        # Fase 1+2: Trascrizione audio
        model = get_model()
        predicted_notes = transcribe_audio(str(audio_path), model)

        # Caricamento spartito
        ref_suffix = ref_path.suffix.lower()
        if ref_suffix in (".mid", ".midi"):
            ref_annotations = parse_midi(str(ref_path))
        elif ref_suffix == ".jams":
            ref_annotations = parse_jams(str(ref_path))
        else:
            raise HTTPException(400, f"Formato non supportato: {ref_suffix}")

        reference_notes = build_note_sequence(ref_annotations)

        # Fase 3: Allineamento DTW
        error_log = run_alignment(predicted_notes, reference_notes, time_tolerance)

        # Fase 4: Feedback LLM
        feedback_text = None
        feedback_error = None

        if generate_feedback_flag:
            try:
                feedback_text = generate_feedback(error_log, context)
            except Exception as e:
                logger.error(f"Errore generazione feedback: {e}")
                feedback_error = str(e)

        return {
            "predicted_notes": predicted_notes,
            "reference_notes": reference_notes,
            "error_log": error_log,
            "feedback": feedback_text,
            "feedback_error": feedback_error,
        }

    except FileNotFoundError:
        raise HTTPException(
            503,
            "Modello non disponibile. Assicurati che i pesi siano in "
            f"{config.WEIGHTS_DIR}/{config.WEIGHTS_FILENAME}"
        )
    except Exception as e:
        logger.error(f"Errore nell'analisi: {e}", exc_info=True)
        raise HTTPException(500, f"Errore nell'analisi: {str(e)}")
    finally:
        audio_path.unlink(missing_ok=True)
        ref_path.unlink(missing_ok=True)


@app.get("/api/config")
async def get_config():
    """Restituisce la configurazione corrente della pipeline."""
    return {
        "sample_rate": config.SAMPLE_RATE,
        "hop_length": config.HOP_LENGTH,
        "n_bins": config.N_BINS,
        "time_tolerance": config.TIME_TOLERANCE,
        "openai_model": config.OPENAI_MODEL,
        "weights_path": str(config.WEIGHTS_PATH),
        "weights_available": config.WEIGHTS_PATH.exists(),
        "api_key_configured": bool(config.OPENAI_API_KEY),
    }
