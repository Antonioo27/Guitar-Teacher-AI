"""
pipeline.py — Orchestrazione end-to-end della pipeline AI Guitar Tutor.

Classe principale che collega tutti i moduli:
  Modulo 1 (audio_processing) → Modulo 2 (model + inference) →
  Modulo 3 (alignment) → Modulo 4 (feedback)

Fornisce anche un entry point CLI per l'esecuzione da terminale.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from . import config
from .model import load_model, TabCNN
from .inference import transcribe_audio
from .alignment import run_alignment
from .feedback import generate_feedback
from .dataset import parse_midi, parse_jams, build_note_sequence

logger = logging.getLogger(__name__)


class GuitarTutorPipeline:
    """
    Pipeline completa dell'AI Guitar Tutor.

    Orchestrazione sequenziale:
    1. Caricamento del modello TabCNN (una volta sola)
    2. Trascrizione dell'audio dello studente (Moduli 1+2)
    3. Caricamento dello spartito di riferimento
    4. Allineamento DTW e classificazione errori (Modulo 3)
    5. Generazione feedback con LLM (Modulo 4)

    Usage:
        pipeline = GuitarTutorPipeline()
        result = pipeline.run(
            audio_path="student_recording.wav",
            reference_path="exercise.mid",
            exercise_context="Scala di Do Maggiore a 120 BPM"
        )
        print(result["feedback"])
    """

    def __init__(
        self,
        weights_path: str | Path = config.WEIGHTS_PATH,
        device: str | None = None,
    ):
        """
        Inizializza la pipeline caricando il modello TabCNN.

        Args:
            weights_path: Percorso ai pesi del modello.
            device: Device PyTorch. Se None, usa CUDA se disponibile.
        """
        import torch

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Inizializzazione pipeline su device: {self.device}")

        # Carica il modello TabCNN
        self.model = load_model(weights_path, self.device)
        logger.info("Modello TabCNN caricato con successo.")

    def _load_reference(self, reference_path: str) -> list[dict[str, Any]]:
        """
        Carica lo spartito di riferimento da file MIDI o JAMS.

        Args:
            reference_path: Percorso al file di riferimento (.mid, .midi, .jams).

        Returns:
            Sequenza di note di riferimento.

        Raises:
            ValueError: Se il formato del file non è supportato.
        """
        path = Path(reference_path)
        suffix = path.suffix.lower()

        if suffix in (".mid", ".midi"):
            annotations = parse_midi(reference_path)
        elif suffix == ".jams":
            annotations = parse_jams(reference_path)
        else:
            raise ValueError(
                f"Formato non supportato: '{suffix}'. "
                f"Usa file .mid, .midi o .jams"
            )

        return build_note_sequence(annotations)

    def run(
        self,
        audio_path: str,
        reference_path: str,
        exercise_context: str = "",
        time_tolerance: float = config.TIME_TOLERANCE,
        generate_llm_feedback: bool = True,
    ) -> dict[str, Any]:
        """
        Esegue l'intera pipeline AI Guitar Tutor.

        Args:
            audio_path: Percorso al file audio dello studente (.wav).
            reference_path: Percorso allo spartito di riferimento (.mid/.jams).
            exercise_context: Descrizione dell'esercizio per il feedback LLM.
            time_tolerance: Tolleranza temporale per il DTW (secondi).
            generate_llm_feedback: Se True, genera il feedback con LLM.

        Returns:
            Dizionario con i risultati di ogni fase:
            - "predicted_notes": note trascritte dalla TabCNN
            - "reference_notes": note dallo spartito
            - "error_log": report errori dal DTW
            - "feedback": testo del feedback LLM (se richiesto)
        """
        logger.info("=" * 60)
        logger.info("AI GUITAR TUTOR — Pipeline Start")
        logger.info("=" * 60)

        result = {}

        # =====================================================================
        # Fase 1+2: Trascrizione audio → note predette
        # =====================================================================
        logger.info("[Modulo 1+2] Trascrizione audio...")
        predicted_notes = transcribe_audio(audio_path, self.model, self.device)
        result["predicted_notes"] = predicted_notes
        logger.info(f"  → {len(predicted_notes)} note trascritte")

        # =====================================================================
        # Caricamento spartito di riferimento
        # =====================================================================
        logger.info("[Ref] Caricamento spartito di riferimento...")
        reference_notes = self._load_reference(reference_path)
        result["reference_notes"] = reference_notes
        logger.info(f"  → {len(reference_notes)} note nello spartito")

        # =====================================================================
        # Fase 3: Allineamento DTW e classificazione errori
        # =====================================================================
        logger.info("[Modulo 3] Allineamento DTW...")
        error_log = run_alignment(predicted_notes, reference_notes, time_tolerance)
        result["error_log"] = error_log
        logger.info(
            f"  → Accuracy: {error_log['summary']['accuracy_percent']}%"
        )

        # =====================================================================
        # Fase 4: Generazione feedback LLM
        # =====================================================================
        if generate_llm_feedback:
            logger.info("[Modulo 4] Generazione feedback LLM...")
            try:
                feedback = generate_feedback(error_log, exercise_context)
                result["feedback"] = feedback
                logger.info("  → Feedback generato con successo")
            except Exception as e:
                logger.error(f"  → Errore nella generazione del feedback: {e}")
                result["feedback"] = None
                result["feedback_error"] = str(e)
        else:
            result["feedback"] = None

        logger.info("=" * 60)
        logger.info("AI GUITAR TUTOR — Pipeline Complete")
        logger.info("=" * 60)

        return result


# =============================================================================
# Entry Point CLI
# =============================================================================

def main():
    """Entry point per l'esecuzione da riga di comando."""
    parser = argparse.ArgumentParser(
        description="AI Guitar Tutor — Trascrizione e valutazione dell'esecuzione chitarristica",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempio d'uso:
  python -m guitar_tutor_pipeline.src.pipeline \\
      --audio recording.wav \\
      --reference exercise.mid \\
      --context "Scala di Do Maggiore a 120 BPM"
        """,
    )
    parser.add_argument(
        "--audio", "-a",
        required=True,
        help="Percorso al file audio dell'esecuzione dello studente (.wav)",
    )
    parser.add_argument(
        "--reference", "-r",
        required=True,
        help="Percorso allo spartito di riferimento (.mid, .midi, .jams)",
    )
    parser.add_argument(
        "--context", "-c",
        default="",
        help="Descrizione dell'esercizio (es. 'Scala di Do Maggiore a 120 BPM')",
    )
    parser.add_argument(
        "--weights", "-w",
        default=str(config.WEIGHTS_PATH),
        help=f"Percorso ai pesi del modello (default: {config.WEIGHTS_PATH})",
    )
    parser.add_argument(
        "--device", "-d",
        default=None,
        choices=["cpu", "cuda"],
        help="Device di calcolo (default: auto-detect)",
    )
    parser.add_argument(
        "--tolerance", "-t",
        type=float,
        default=config.TIME_TOLERANCE,
        help=f"Tolleranza temporale in secondi (default: {config.TIME_TOLERANCE})",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Disabilita la generazione del feedback LLM",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Percorso per salvare il report JSON (default: stampa a video)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Abilita output verboso (debug logging)",
    )

    args = parser.parse_args()

    # Configurazione logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        # Inizializza ed esegui la pipeline
        pipeline = GuitarTutorPipeline(
            weights_path=args.weights,
            device=args.device,
        )

        result = pipeline.run(
            audio_path=args.audio,
            reference_path=args.reference,
            exercise_context=args.context,
            time_tolerance=args.tolerance,
            generate_llm_feedback=not args.no_feedback,
        )

        # Output dei risultati
        output_data = {
            "summary": result["error_log"]["summary"],
            "errors": result["error_log"]["errors"],
            "num_predicted_notes": len(result["predicted_notes"]),
            "num_reference_notes": len(result["reference_notes"]),
        }

        if result.get("feedback"):
            output_data["feedback"] = result["feedback"]

        if args.output:
            # Salva su file
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Report salvato in: {output_path}")
        else:
            # Stampa a video
            print("\n" + "=" * 60)
            print("RISULTATI DELL'ANALISI")
            print("=" * 60)
            print(json.dumps(output_data["summary"], indent=2))

            if result.get("feedback"):
                print("\n" + "=" * 60)
                print("FEEDBACK DEL TUTOR")
                print("=" * 60)
                print(result["feedback"])

    except FileNotFoundError as e:
        logger.error(f"File non trovato: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Errore: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
