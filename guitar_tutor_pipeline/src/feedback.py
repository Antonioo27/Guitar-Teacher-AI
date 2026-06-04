"""
feedback.py — Modulo 4: Generazione del Feedback (Il Tutor - Core AI 2).

Trasforma il log degli errori algoritmico prodotto dal DTW (Modulo 3)
in un feedback testuale empatico e pedagogico per lo studente di chitarra,
usando un Large Language Model (OpenAI API).

Il prompt è strutturato per ottenere un feedback che:
- Riconosca i punti di forza dell'esecuzione
- Identifichi i pattern di errore (non solo i singoli errori)
- Suggerisca esercizi mirati per migliorare
- Usi un tono incoraggiante e motivante
"""

import json
import logging
from typing import Any

from openai import OpenAI

from . import config

logger = logging.getLogger(__name__)


# =============================================================================
# System Prompt per il Tutor
# =============================================================================

SYSTEM_PROMPT = """Sei un insegnante di chitarra esperto, paziente e incoraggiante.
Il tuo compito è analizzare l'esecuzione di uno studente e fornire un feedback
costruttivo e pedagogico in italiano.

REGOLE:
1. Inizia SEMPRE riconoscendo qualcosa di positivo nell'esecuzione.
2. Identifica PATTERN negli errori, non elencarli uno per uno.
3. Usa un linguaggio semplice e musicale (non tecnico/informatico).
4. Suggerisci 2-3 esercizi SPECIFICI per migliorare i punti deboli.
5. Concludi con una frase motivante.
6. Mantieni il feedback conciso (massimo 300 parole).

CATEGORIE DI ERRORE:
- "correct": nota giusta al momento giusto
- "wrong_timing": nota giusta ma fuori tempo (il ritmo è il problema)
- "wrong_pitch": nota sbagliata (la diteggiatura è il problema)
- "missing": nota dello spartito non suonata (omissione)
- "extra": nota suonata in più (nota spuria o esitazione)

Quando parli delle note, usa i nomi italiani:
C=DO, D=RE, E=MI, F=FA, G=SOL, A=LA, B=SI"""


# =============================================================================
# Costruzione del prompt
# =============================================================================

def build_prompt(
    error_log: dict[str, Any],
    context: str = "",
) -> str:
    """
    Costruisce il prompt utente con il contesto dell'esercizio e il
    JSON degli errori da inviare all'LLM.

    Args:
        error_log: Report degli errori dal Modulo 3 (alignment.py).
        context: Descrizione dell'esercizio (es. "Scala di Do Maggiore a 120 BPM").

    Returns:
        Prompt formattato per l'LLM.
    """
    prompt_parts = []

    if context:
        prompt_parts.append(f"ESERCIZIO: {context}")

    prompt_parts.append(f"RIEPILOGO ESECUZIONE:\n{json.dumps(error_log['summary'], indent=2)}")

    if error_log["errors"]:
        # Limita il numero di errori nel prompt per non superare il contesto
        max_errors = 30
        errors_to_show = error_log["errors"][:max_errors]

        # Semplifica il formato per il prompt
        simplified_errors = []
        for err in errors_to_show:
            simplified = {
                "tempo": err["time"],
                "atteso": err.get("expected"),
                "suonato": err.get("played"),
                "status": err["status"],
            }
            if err.get("delta_t") is not None:
                simplified["delta_t_sec"] = err["delta_t"]
            simplified_errors.append(simplified)

        prompt_parts.append(
            f"DETTAGLIO ERRORI (primi {len(errors_to_show)}):\n"
            f"{json.dumps(simplified_errors, indent=2, ensure_ascii=False)}"
        )

        if len(error_log["errors"]) > max_errors:
            prompt_parts.append(
                f"(... e altri {len(error_log['errors']) - max_errors} errori omessi)"
            )

    prompt_parts.append(
        "Analizza l'esecuzione e fornisci un feedback pedagogico allo studente."
    )

    return "\n\n".join(prompt_parts)


# =============================================================================
# Chiamata al LLM
# =============================================================================

def call_openai(
    user_prompt: str,
    model: str = config.OPENAI_MODEL,
    temperature: float = config.LLM_TEMPERATURE,
    max_tokens: int = config.LLM_MAX_TOKENS,
) -> str:
    """
    Chiama l'API OpenAI per generare il feedback testuale.

    Richiede che la variabile d'ambiente OPENAI_API_KEY sia configurata
    (nel file .env o nelle variabili di sistema).

    Args:
        user_prompt: Prompt con contesto e errori.
        model: Nome del modello OpenAI (default: gpt-4o-mini).
        temperature: Creatività della risposta (0.0 = deterministico).
        max_tokens: Numero massimo di token nella risposta.

    Returns:
        Testo del feedback generato dall'LLM.

    Raises:
        ValueError: Se la API key non è configurata.
        openai.APIError: Se la chiamata API fallisce.
    """
    if not config.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY non configurata. "
            "Crea un file .env nella root del progetto con:\n"
            "OPENAI_API_KEY=sk-..."
        )

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    logger.info(f"Chiamata a OpenAI ({model})...")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    feedback = response.choices[0].message.content
    logger.info(f"Feedback generato ({len(feedback)} caratteri)")

    return feedback


# =============================================================================
# Funzione di alto livello
# =============================================================================

def generate_feedback(
    error_log: dict[str, Any],
    context: str = "",
) -> str:
    """
    Genera il feedback didattico completo a partire dal report errori.

    Questa è la funzione principale del Modulo 4, che orchestra la
    costruzione del prompt e la chiamata all'LLM.

    Args:
        error_log: Report errori dal Modulo 3 (vedi alignment.build_error_log).
        context: Descrizione dell'esercizio (es. "Scala di Do Maggiore a 120 BPM").

    Returns:
        Testo del feedback pedagogico in italiano.
    """
    logger.info("Generazione feedback con LLM...")

    # 1. Costruisci il prompt
    user_prompt = build_prompt(error_log, context)

    # 2. Chiama l'LLM
    feedback = call_openai(user_prompt)

    return feedback
