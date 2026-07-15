"""
api.py — REST API per il frontend Vue dell'AI Guitar Tutor.

Espone la pipeline come servizio web con FastAPI:
- Upload file audio (.wav)
- Upload spartito di riferimento (.mid/.jams)
- Selezione del modello di trascrizione (TabCNN / CRNN)
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
from typing import Optional, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .src.app import config
from .src.app.model_registry import (
    load_model,
    transcribe,
    get_available_models,
)
from .src.app.alignment import run_alignment
from .src.app.feedback import generate_feedback
from .src.app.dataset import parse_midi, parse_jams, build_note_sequence

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
    version="2.0.0",
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
# Gestione del modello (caricamento dinamico)
# =========================================================================
_model: Optional[Any] = None
_current_model_name: str = config.DEFAULT_MODEL


class ModelSelection(BaseModel):
    """Schema per la selezione del modello."""
    model: str


def get_model() -> tuple[Any, str]:
    """Lazy-load del modello corrente."""
    global _model, _current_model_name
    if _model is None:
        try:
            _model = load_model(_current_model_name)
            logger.info(f"Modello {_current_model_name} caricato con successo.")
        except FileNotFoundError as e:
            logger.error(f"Pesi del modello non trovati: {e}")
            raise
    return _model, _current_model_name


def _switch_model(model_name: str) -> None:
    """Cambia il modello attivo, scaricando quello precedente."""
    global _model, _current_model_name

    if model_name not in config.AVAILABLE_MODELS:
        raise ValueError(
            f"Modello '{model_name}' non riconosciuto. "
            f"Disponibili: {config.AVAILABLE_MODELS}"
        )

    if model_name == _current_model_name and _model is not None:
        logger.info(f"Modello {model_name} già attivo.")
        return

    # Scarica il vecchio modello dalla memoria
    if _model is not None:
        logger.info(f"Scaricamento modello {_current_model_name}...")
        del _model
        _model = None

        # Libera la memoria GPU se possibile
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    _current_model_name = model_name
    # Il modello verrà caricato al prossimo get_model()
    logger.info(f"Modello selezionato: {model_name} (verrà caricato alla prossima richiesta)")


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
# Endpoints — Gestione Modello
# =========================================================================

@app.get("/api/models")
async def list_models():
    """
    Restituisce la lista dei modelli disponibili e il modello attualmente attivo.
    """
    models = get_available_models()
    return {
        "models": models,
        "current_model": _current_model_name,
    }


@app.get("/api/model")
async def get_current_model():
    """Restituisce il modello attualmente selezionato."""
    model_loaded = _model is not None
    return {
        "current_model": _current_model_name,
        "model_loaded": model_loaded,
    }


@app.post("/api/model")
async def set_model(selection: ModelSelection):
    """
    Cambia il modello di trascrizione attivo.

    Body JSON: {"model": "TabCNN"} oppure {"model": "CRNN"}
    """
    try:
        _switch_model(selection.model)
        return {
            "status": "ok",
            "current_model": _current_model_name,
            "message": f"Modello cambiato a {_current_model_name}",
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


# =========================================================================
# Endpoints — Salute e Configurazione
# =========================================================================

@app.get("/api/health")
async def health_check():
    """Verifica che il server sia attivo."""
    model_loaded = _model is not None
    # weights_available riflette il modello ATTIVO: è ciò che lo StatusBar del
    # frontend legge per il badge di stato. Prima mancava qui (era solo in
    # /api/config) e il badge mostrava sempre "Pesi mancanti".
    if _current_model_name == "CRNN":
        weights_available = config.CRNN_WEIGHTS_PATH.exists()
    else:
        weights_available = config.WEIGHTS_PATH.exists()
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "current_model": _current_model_name,
        "available_models": config.AVAILABLE_MODELS,
        "weights_available": weights_available,
        "tabcnn_weights_available": config.WEIGHTS_PATH.exists(),
        "crnn_weights_available": config.CRNN_WEIGHTS_PATH.exists(),
    }


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
        "crnn_weights_path": str(config.CRNN_WEIGHTS_PATH),
        "crnn_weights_available": config.CRNN_WEIGHTS_PATH.exists(),
        "api_key_configured": bool(config.OPENAI_API_KEY),
        "current_model": _current_model_name,
        "available_models": config.AVAILABLE_MODELS,
    }


# =========================================================================
# Endpoints — Trascrizione e Analisi
# =========================================================================

@app.post("/api/transcribe")
async def transcribe_endpoint(
    audio: UploadFile = File(..., description="File audio .wav dello studente"),
):
    """
    Moduli 1+2: Trascrivi un file audio in una sequenza di note.
    Restituisce le note predette senza confronto con lo spartito.
    Usa il modello attualmente selezionato.
    """
    if not audio.filename.lower().endswith((".wav", ".mp3", ".flac")):
        raise HTTPException(400, "Formato audio non supportato. Usa .wav, .mp3 o .flac")

    audio_path = _save_upload(audio, ".wav")

    try:
        model, model_name = get_model()
        notes = transcribe(str(audio_path), model, model_name)
        return {
            "notes": notes,
            "total_notes": len(notes),
            "model_used": model_name,
        }
    except FileNotFoundError:
        raise HTTPException(
            503,
            f"Modello {_current_model_name} non disponibile. "
            "Assicurati che i pesi siano presenti."
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
    Pipeline completa: trascrizione + allineamento + feedback LLM.
    Usa il modello attualmente selezionato.

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
        model, model_name = get_model()
        predicted_notes = transcribe(str(audio_path), model, model_name)

        # Caricamento spartito
        ref_suffix = ref_path.suffix.lower()
        if ref_suffix in (".mid", ".midi"):
            ref_annotations = parse_midi(str(ref_path))
        elif ref_suffix == ".jams":
            ref_annotations = parse_jams(str(ref_path))
        else:
            raise HTTPException(400, f"Formato non supportato: {ref_suffix}")

        reference_notes = build_note_sequence(ref_annotations)

        # Fase 3: Allineamento
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
            "model_used": model_name,
        }

    except FileNotFoundError:
        raise HTTPException(
            503,
            f"Modello {_current_model_name} non disponibile. "
            "Assicurati che i pesi siano presenti."
        )
    except Exception as e:
        logger.error(f"Errore nell'analisi: {e}", exc_info=True)
        raise HTTPException(500, f"Errore nell'analisi: {str(e)}")
    finally:
        audio_path.unlink(missing_ok=True)
        ref_path.unlink(missing_ok=True)
