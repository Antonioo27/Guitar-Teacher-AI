"""
runTabCNN.py — Script standalone per trascrivere una matrice CQT (.npy) con TabCNN.

Questo script esegue l'inferenza del modello definito in model.py a partire
NON dall'audio grezzo, ma dalla matrice CQT già normalizzata in decibel
prodotta da convertWawToCQT.py:

    <nome>_cqt.npy  →  tensore  →  TabCNN  →  predizioni frame-by-frame  →  note

In questo modo possiamo procedere per step: prima si genera la CQT con
convertWawToCQT.py, poi la si dà in pasto al modello con questo script,
variando eventualmente i parametri della CQT a monte.

La matrice .npy attesa ha:
    - shape (N_BINS, T)  →  (192, n_frame)
    - valori normalizzati in [0, 1]  (= (dB / 80) + 1)

Output salvati (opzionali, con --output_dir):
    - <nome>_notes.json   → lista di note in formato JSON
    - <nome>_notes.csv    → stessa lista in formato tabellare

Uso:
    python -m guitar_tutor_pipeline.src.runTabCNN output_cqt/00_CMaj_120_cqt.npy
    python -m guitar_tutor_pipeline.src.runTabCNN matrice.npy --output_dir output_notes/
    python -m guitar_tutor_pipeline.src.runTabCNN matrice.npy --device cuda
"""

import sys
import csv
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import torch

from . import config
from .model import load_model
from .inference import decode_predictions

# =========================================================================
# Configurazione del logger — coerente con convertWawToCQT.py
# =========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =========================================================================
# FASE A — Caricamento della matrice CQT dal file .npy
# =========================================================================
def load_cqt_matrix(npy_path: str) -> np.ndarray:
    """
    Carica la matrice CQT normalizzata da un file .npy su disco.

    Si aspetta una matrice 2D (N_BINS, T) con valori in [0, 1], cioè
    esattamente l'output di convertWawToCQT.py.

    Returns:
        cqt_norm: array numpy (192, T) di tipo float32
    """
    logger.info("=" * 70)
    logger.info("FASE A — Caricamento matrice CQT (.npy)")
    logger.info("=" * 70)

    path = Path(npy_path)
    if not path.exists():
        logger.error(f"File .npy non trovato: {npy_path}")
        sys.exit(1)

    cqt_norm = np.load(path)

    # Validazione della forma: ci aspettiamo (N_BINS, T)
    if cqt_norm.ndim != 2:
        logger.error(
            f"Matrice con {cqt_norm.ndim} dimensioni: attesa una matrice 2D (n_bins, n_frame)."
        )
        sys.exit(1)

    if cqt_norm.shape[0] != config.N_BINS:
        logger.warning(
            f"  Attenzione: la matrice ha {cqt_norm.shape[0]} bin, "
            f"ma il modello ne attende {config.N_BINS}. "
            f"Probabile mismatch dei parametri CQT (n_bins)."
        )

    # Assicura il tipo corretto per PyTorch
    cqt_norm = cqt_norm.astype(np.float32)

    logger.info(f"  File:        {path.name}")
    logger.info(f"  Shape:       {cqt_norm.shape}  →  ({cqt_norm.shape[0]} bin × {cqt_norm.shape[1]} frame)")
    logger.info(f"  Dtype:       {cqt_norm.dtype}")
    logger.info(f"  Range:       [{cqt_norm.min():.4f}, {cqt_norm.max():.4f}]")
    if cqt_norm.min() < -0.01 or cqt_norm.max() > 1.01:
        logger.warning("  I valori escono da [0, 1]: la matrice potrebbe non essere normalizzata.")

    return cqt_norm


# =========================================================================
# FASE B — Costruzione del tensore di input per il modello
# =========================================================================
def build_input_tensor(cqt_norm: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """
    Converte la matrice CQT (192, T) nel tensore atteso da TabCNN.

    Il modello (tramite pre_proc) si aspetta un batch con la feature map
    nella forma (Batch=1, Freq=192, Time=T). Il framing locale a finestre
    di CONTEXT_FRAMES è poi gestito internamente da pre_proc().

    Returns:
        tensor: torch.Tensor di shape (1, 192, T) sul device richiesto
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("FASE B — Costruzione tensore di input")
    logger.info("=" * 70)

    # (192, T) → (1, 192, T): aggiunge la dimensione di batch
    tensor = torch.tensor(cqt_norm, dtype=torch.float32).unsqueeze(0).to(device)

    logger.info(f"  Tensor shape: {tuple(tensor.shape)}  (Batch=1, Freq, Time)")
    logger.info(f"  Device:       {tensor.device}")

    return tensor


# =========================================================================
# FASE C — Esecuzione del modello (forward + estrazione confidenze)
# =========================================================================
def run_model(model, input_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    Esegue l'inferenza del modello TabCNN sul tensore di input.

    Passaggi (replicati qui invece di usare inference.transcribe_audio):
        1. pre_proc   — framing locale a finestre di contesto
        2. forward    — produce la distribuzione softmax per (frame, corda, classe)
        3. confidenze — max sulla distribuzione softmax = quanto è sicuro il modello
        4. post_proc  — argmax → tasto predetto per ogni (frame, corda)

    Returns:
        predictions: matrice (T, 6) con il fret predetto per frame e corda
        confidences: matrice (T, 6) con la confidenza [0,1] di ogni predizione
    """
    from amt_tools import tools

    logger.info("")
    logger.info("=" * 70)
    logger.info("FASE C — Inferenza TabCNN")
    logger.info("=" * 70)

    batch = {tools.KEY_FEATS: input_tensor}

    with torch.no_grad():
        # 1. Pre-processing interno del modello (framing locale)
        batch = model.pre_proc(batch)

        # 2. Forward → distribuzione softmax grezza
        output = model(batch[tools.KEY_FEATS])
        batch[tools.KEY_OUTPUT] = output

        # 3. Estrazione confidenze
        #    ATTENZIONE: output[KEY_TABLATURE] sono LOGIT grezzi (output di un
        #    nn.Linear), NON probabilità. La softmax la applica il modello solo
        #    dentro finalize_output. Quindi qui dobbiamo applicarla noi sulle
        #    21 classi per ottenere una confidenza reale in [0, 1].
        raw_logits = output[tools.KEY_TABLATURE]
        raw_np = (
            raw_logits.cpu().numpy()
            if isinstance(raw_logits, torch.Tensor)
            else np.array(raw_logits)
        )

        # Rimuove le dimensioni fittizie di lunghezza 1 (es. (705, 1, 126) → (705, 126))
        raw_np = np.squeeze(raw_np)

        # Porta i logit alla forma (T, 6, 21): un gruppo softmax per corda
        if raw_np.ndim == 2:
            # Caso "schiacciato" (T, 126) → separa gruppi (6) × classi (21)
            T_frames = raw_np.shape[0]
            logits = raw_np.reshape(T_frames, 6, -1)
        elif raw_np.ndim == 3:
            # Caso già separato (T, 6, 21)
            logits = raw_np
        else:
            logger.warning(f"Shape logit inattesa {raw_np.shape}: confidenze = 1.0")
            logits = None

        if logits is not None:
            # Softmax numericamente stabile sull'asse delle classi
            shifted = logits - logits.max(axis=-1, keepdims=True)
            exp = np.exp(shifted)
            probs = exp / exp.sum(axis=-1, keepdims=True)
            # Confidenza = probabilità della classe vincente, per ogni (frame, corda)
            confidences = probs.max(axis=-1)  # (T, 6)
        else:
            confidences = None

        # 4. Post-processing → argmax (tasto predetto)
        risultato = model.post_proc(batch)

    # Estrazione matrice fret per frame → forma (T, 6)
    tab = risultato[tools.KEY_TABLATURE]
    tab_np = tab.cpu().numpy() if isinstance(tab, torch.Tensor) else np.asarray(tab)
    tab_np = np.squeeze(tab_np)
    if tab_np.ndim == 1:
        tab_np = tab_np.reshape(-1, 6)
    elif tab_np.ndim == 3 and tab_np.shape[1] == 6:
        tab_np = tab_np[:, :, 0]
    predictions = tab_np

    # Fallback confidenze se l'estrazione non combacia con le predizioni
    if confidences is None or confidences.shape != predictions.shape:
        logger.warning(
            f"Confidenze non disponibili/compatibili "
            f"({getattr(confidences, 'shape', None)} vs {predictions.shape}): uso 1.0."
        )
        confidences = np.ones_like(predictions, dtype=np.float32)

    logger.info(f"  Predizioni shape: {predictions.shape}  (frame × 6 corde)")
    logger.info(
        f"  Confidenze:       media={confidences.mean():.3f}  "
        f"min={confidences.min():.3f}  max={confidences.max():.3f}"
    )

    return predictions, confidences


# =========================================================================
# Rilevamento onset dalla CQT (per distinguere ribattuti da note sostenute)
# =========================================================================
def detect_onset_times(
    cqt_norm: np.ndarray,
    hop_length: int,
    sr: int,
    delta: float = 0.07,
) -> np.ndarray:
    """
    Rileva gli istanti di attacco (onset) direttamente dalla matrice CQT.

    Idea: un nuovo attacco (pizzicata) produce un aumento improvviso di
    energia nello spettro → spectral flux = somma delle differenze POSITIVE
    tra frame consecutivi. Una nota sostenuta che il modello "perde" non ha
    questo picco. Usiamo il flux come inviluppo di onset e ne estraiamo i
    picchi con il peak-picker di librosa.

    Args:
        cqt_norm:   matrice (n_bins, T) normalizzata, la stessa data al modello.
        hop_length: hop usato per generare la CQT (per convertire frame → secondi).
        sr:         sample rate usato per generare la CQT.
        delta:      soglia di prominenza del picco (più alto = meno onset).

    Returns:
        onset_times: array di istanti (secondi) in cui c'è un attacco.
    """
    import librosa

    # Spectral flux: solo le variazioni positive (l'energia che "sale")
    flux = np.maximum(np.diff(cqt_norm, axis=1), 0.0).sum(axis=0)
    if flux.max() > 0:
        flux = flux / flux.max()

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=flux,
        sr=sr,
        hop_length=hop_length,
        delta=delta,
        backtrack=False,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    logger.info(f"  Onset rilevati dalla CQT: {len(onset_times)}")
    return onset_times


# =========================================================================
# Risoluzione delle "ghost note" + frammenti di note sostenute
# =========================================================================
def resolve_ghost_notes(
    notes: list[dict],
    max_gap_s: float = 0.06,
    onset_times: np.ndarray | None = None,
    onset_tol_s: float = 0.06,
) -> list[dict]:
    """
    Fonde i frammenti che appartengono alla stessa nota fisica, risolvendo
    due problemi insieme:

    1. Ghost note (collisione inter-corda)
       Lo stesso pitch acceso su più corde dalle 6 softmax indipendenti di
       TabCNN. Una nota pizzicata vive su UNA sola corda.

    2. Note sostenute spezzate (dropout del modello)
       Una nota lunga in cui la confidenza cala a metà, generando due
       frammenti separati da un gap.

    Strategia (onset-aware)
    -----------------------
    Raggruppa per PITCH (ignorando la corda). Due frammenti consecutivi dello
    stesso pitch vengono fusi se sono lo stesso evento fisico. Il criterio NON
    è (solo) la distanza temporale, ma la presenza di un ATTACCO nell'INTERVALLO
    DEL GAP tra i due frammenti:

      - se NESSUN onset cade nel gap [fine_cluster, inizio_frammento] → è la
        stessa nota sostenuta/ghost → FONDI (anche con gap ampio);
      - se un onset cade nel gap → è un ri-pizzicato → tieni SEPARATO.

    Perché l'intervallo e non il solo inizio del frammento: il decoder spesso
    apre il frammento successivo in RITARDO rispetto all'attacco reale (anche
    150-200ms), quindi un controllo puntuale sull'inizio mancherebbe l'onset.
    Cercare l'onset in tutto il gap cattura il ri-pizzicato anche se il modello
    è in ritardo.

    Se onset_times è None si ricade sul solo criterio di gap (max_gap_s).
    La nota risultante copre l'intervallo unione ed eredita corda/tasto/
    confidenza dal frammento con la CONFIDENZA DI PICCO più alta.

    Args:
        notes:       note decodificate.
        max_gap_s:   gap di anti-chattering: due frammenti più vicini di questo
                     vengono fusi comunque (filtra il flicker del decoder). È
                     anche l'unico criterio quando non ci sono onset.
        onset_times: istanti di attacco (secondi) rilevati dalla CQT.
        onset_tol_s: tolleranza con cui si allarga l'intervallo del gap quando
                     si cerca un onset.

    Returns:
        Lista di note fuse, riordinata per onset.
    """
    if not notes:
        return notes

    have_onsets = onset_times is not None and len(onset_times) > 0

    def _onset_in_gap(t0: float, t1: float) -> bool:
        """C'è un attacco rilevato nell'intervallo [t0, t1] (con tolleranza)?"""
        if not have_onsets:
            return False
        return bool(np.any((onset_times >= t0 - onset_tol_s) & (onset_times <= t1 + onset_tol_s)))

    from collections import defaultdict
    by_pitch: dict[int, list[dict]] = defaultdict(list)
    for n in notes:
        by_pitch[n["pitch"]].append(n)

    def _flush(cluster: list[dict]) -> dict:
        # Rappresentante = frammento con confidenza di picco massima
        best = max(cluster, key=lambda x: x.get("confidence", 0.0))
        start = min(c["time"] for c in cluster)
        end = max(c["time"] + c["duration"] for c in cluster)
        merged = dict(best)
        merged["time"] = round(start, 4)
        merged["duration"] = round(end - start, 4)
        return merged

    resolved: list[dict] = []
    for pitch, group in by_pitch.items():
        group.sort(key=lambda x: x["time"])

        cluster = [group[0]]
        cluster_end = group[0]["time"] + group[0]["duration"]

        for nxt in group[1:]:
            gap = nxt["time"] - cluster_end
            if have_onsets:
                # Fondi se NON c'è un attacco nel gap (nota sostenuta/ghost)
                # OPPURE se il gap è sotto la soglia di anti-chattering.
                same_note = (not _onset_in_gap(cluster_end, nxt["time"])) or (gap <= max_gap_s)
            else:
                # Senza onset: ricade sul solo criterio di gap.
                same_note = gap <= max_gap_s

            if same_note:
                cluster.append(nxt)
                cluster_end = max(cluster_end, nxt["time"] + nxt["duration"])
            else:
                resolved.append(_flush(cluster))
                cluster = [nxt]
                cluster_end = nxt["time"] + nxt["duration"]

        resolved.append(_flush(cluster))

    resolved.sort(key=lambda x: (x["time"], x.get("string", -1)))
    return resolved


# =========================================================================
# Filtro durata minima — rimuove frammenti spuri troppo corti
# =========================================================================
def filter_short_notes(notes: list[dict], min_duration_s: float = 0.12) -> list[dict]:
    """
    Scarta le note più corte di min_duration_s.

    Dopo la fusione onset-aware rimangono a volte micro-frammenti (<120ms)
    generati da artefatti del modello (echi spettrali, misclassificazioni
    brevi). Sono tipicamente più corti di qualsiasi nota realmente suonata,
    quindi un filtro di durata li elimina senza toccare le note vere.
    """
    if min_duration_s <= 0:
        return notes
    before = len(notes)
    result = [n for n in notes if n.get("duration", 0.0) >= min_duration_s]
    removed = before - len(result)
    if removed:
        logger.info(
            f"  Filtro durata (>={min_duration_s*1000:.0f}ms): "
            f"{before} → {len(result)} note (scartate {removed})"
        )
    return result


# =========================================================================
# Stampa a schermo della tablatura predetta
# =========================================================================
def print_notes_table(notes: list[dict]) -> None:
    """
    Mostra le note trascritte in una tabella leggibile nel terminale.
    """
    if not notes:
        logger.warning("  Nessuna nota trascritta (matrice silenziosa o sotto soglia).")
        return

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"NOTE TRASCRITTE — {len(notes)} note")
    logger.info("=" * 70)

    header = f"  {'#':>3}  {'t(s)':>7}  {'dur(s)':>7}  {'nota':>5}  {'midi':>4}  {'corda':>5}  {'tasto':>5}  {'conf':>5}"
    logger.info(header)
    logger.info("  " + "-" * (len(header) - 2))

    for i, n in enumerate(notes):
        string_idx = n.get("string", -1)
        string_name = (
            config.STRING_NAMES[string_idx]
            if 0 <= string_idx < len(config.STRING_NAMES)
            else "?"
        )
        logger.info(
            f"  {i:>3}  "
            f"{n.get('time', 0):>7.3f}  "
            f"{n.get('duration', 0):>7.3f}  "
            f"{n.get('note_name', '?'):>5}  "
            f"{n.get('pitch', 0):>4}  "
            f"{string_name:>5}  "
            f"{n.get('fret', 0):>5}  "
            f"{n.get('confidence', 0):>5.2f}"
        )


# =========================================================================
# Salvataggio su disco (JSON + CSV)
# =========================================================================
def save_notes(notes: list[dict], npy_path: str, output_dir: str) -> None:
    """
    Salva la sequenza di note in formato .json e .csv.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(npy_path).stem  # es: "00_CMaj_120_cqt"

    json_path = out_path / f"{stem}_notes.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)
    logger.info(f"  Note JSON salvate: {json_path}")

    csv_path = out_path / f"{stem}_notes.csv"
    fieldnames = ["time", "duration", "pitch", "note_name", "string", "fret", "confidence"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(notes)
    logger.info(f"  Note CSV salvate:  {csv_path}")


# =========================================================================
# MAIN — Entry point per esecuzione standalone
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Trascrive una matrice CQT (.npy) in una sequenza di note usando TabCNN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Esempio:\n"
            "  python -m guitar_tutor_pipeline.src.runTabCNN output_cqt/00_CMaj_120_cqt.npy\n"
            "  python -m guitar_tutor_pipeline.src.runTabCNN matrice.npy --output_dir output_notes/\n"
        ),
    )
    parser.add_argument("npy_path", type=str, help="Percorso al file .npy con la matrice CQT (192, T)")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(config.WEIGHTS_PATH),
        help=f"Percorso ai pesi del modello (default: {config.WEIGHTS_PATH})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device su cui eseguire il modello: cpu o cuda (default: cpu)",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=config.HOP_LENGTH,
        help=f"Hop length usato per generare la CQT (per la temporizzazione note, default: {config.HOP_LENGTH})",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=config.SAMPLE_RATE,
        help=f"Sample rate usato per generare la CQT (per la temporizzazione note, default: {config.SAMPLE_RATE})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Cartella dove salvare le note (.json + .csv). Se omesso, stampa solo a schermo.",
    )
    parser.add_argument(
        "--gap-fill",
        type=float,
        default=0.06,
        help="Gap fill (s) nel decoder: unisce solo il chattering frame-level. "
             "Tenuto basso per non pre-fondere i ri-pizzicati prima della fase onset-aware (default: 0.06).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.12,
        help="Durata minima (s): note più corte vengono scartate come artefatti (default: 0.12).",
    )
    parser.add_argument(
        "--no-ghost-fix",
        action="store_true",
        help="Disattiva la risoluzione delle ghost note (collisioni di pitch tra corde).",
    )
    parser.add_argument(
        "--ghost-gap",
        type=float,
        default=0.06,
        help="Gap di anti-chattering (s) per fondere frammenti dello stesso pitch (default: 0.06).",
    )
    parser.add_argument(
        "--no-onsets",
        action="store_true",
        help="Non usare gli onset della CQT: fonde i frammenti solo in base al gap.",
    )
    parser.add_argument(
        "--onset-delta",
        type=float,
        default=0.07,
        help="Soglia di prominenza per il rilevamento onset dalla CQT (default: 0.07).",
    )
    args = parser.parse_args()

    logger.info("🎸 Trascrizione CQT(.npy) → Note con TabCNN")
    logger.info(f"  Input:   {args.npy_path}")
    logger.info(f"  Pesi:    {args.weights}")
    logger.info(f"  Device:  {args.device}")
    logger.info("")

    # ── FASE A: carica la matrice CQT dal .npy ─────────────────────────
    cqt_norm = load_cqt_matrix(args.npy_path)

    # ── FASE B: costruisci il tensore di input ─────────────────────────
    input_tensor = build_input_tensor(cqt_norm, device=args.device)

    # ── Caricamento del modello ────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("Caricamento modello TabCNN")
    logger.info("=" * 70)
    model = load_model(weights_path=args.weights, device=args.device)

    # ── FASE C: inferenza ──────────────────────────────────────────────
    predictions, confidences = run_model(model, input_tensor)

    # ── Decodifica frame-wise → note (riusa il decoder di inference.py) ─
    # gap_fill ridotto: unisce solo il chattering, lascia intatti i ri-pizzicati
    # così che la fase onset-aware sotto possa decidere se separarli.
    notes = decode_predictions(
        predictions,
        confidences,
        hop_length=args.hop_length,
        sr=args.sr,
        gap_fill_s=args.gap_fill,
    )

    # ── Risoluzione ghost note + frammenti di note sostenute ────────────
    if not args.no_ghost_fix:
        # Rileva gli onset dalla CQT per distinguere ribattuti da note sostenute,
        # a meno che l'utente non abbia disattivato l'uso degli onset.
        onset_times = None
        if not args.no_onsets:
            onset_times = detect_onset_times(
                cqt_norm, hop_length=args.hop_length, sr=args.sr, delta=args.onset_delta
            )

        before = len(notes)
        notes = resolve_ghost_notes(
            notes, max_gap_s=args.ghost_gap, onset_times=onset_times
        )
        logger.info(
            f"  Ghost-fix (gap {args.ghost_gap*1000:.0f}ms, "
            f"onset {'off' if onset_times is None else 'on'}): "
            f"{before} → {len(notes)} note (fuse {before - len(notes)})"
        )

    # ── Filtro durata minima: rimuove i micro-frammenti residui ─────────
    notes = filter_short_notes(notes, min_duration_s=args.min_duration)

    # ── Output ──────────────────────────────────────────────────────────
    print_notes_table(notes)

    if args.output_dir:
        logger.info("")
        logger.info("=" * 70)
        logger.info("SALVATAGGIO RISULTATI")
        logger.info("=" * 70)
        save_notes(notes, args.npy_path, args.output_dir)

    # ── Riepilogo finale ────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ TRASCRIZIONE COMPLETATA")
    logger.info("=" * 70)
    logger.info(f"  Input:        {args.npy_path}")
    logger.info(f"  Note trovate: {len(notes)}")
    if args.output_dir:
        logger.info(f"  Salvate in:   {args.output_dir}/")


if __name__ == "__main__":
    main()
