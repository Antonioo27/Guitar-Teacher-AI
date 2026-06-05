"""
inference.py — Modulo 2: Inferenza (predizione note da audio).

Aggiornato per usare l'implementazione TabCNN del collega tramite amt_tools.
"""

import logging
import numpy as np
import torch
from scipy.ndimage import generic_filter

from . import config
from .audio_processing import prepare_input_tensor
from .dataset import midi_to_note_name, tab_to_midi_pitch
from .model import TabCNN

logger = logging.getLogger(__name__)


def _median_filter_predictions(
    predictions: np.ndarray,
    kernel_size: int = 9,
) -> np.ndarray:
    """
    Median Filter sui frame di predizione (operato PRIMA del decoder).

    Problema che risolve
    --------------------
    TabCNN classifica ogni frame (~11ms) in modo indipendente. In un sustain
    lungo il segnale CQT è quasi costante, ma piccole variazioni causano
    dropout di 1-3 frame in cui il modello predice una classe sbagliata
    (es. "silenzio" o un fret diverso). Il decoder interpreta questi
    dropout come fine nota + inizio nuova nota → chattering.

    Come funziona
    -------------
    Per ogni corda (colonna della matrice T×6) applica una finestra
    scorrevole di `kernel_size` frame e sostituisce il valore centrale
    con la **moda** (valore più frequente) della finestra:

        frame:   [G4, G4, sil, G4, G4, G4, sil, G4]   ← raw
        k=3:     [G4, G4, G4,  G4, G4, G4, G4,  G4]   ← filtrato

    Un kernel da 9 frame copre ~100ms (9 × 11.6ms), sufficiente a
    eliminare dropout brevi senza smussare i veri cambi di nota
    (che tipicamente durano > 200ms).

    Args:
        predictions:  Matrice (T, n_strings) con il fret predetto per frame.
        kernel_size:  Dimensione della finestra (frame). Deve essere dispari.
                      Default 9 ≈ 100ms a HOP=512, SR=44100.

    Returns:
        Matrice (T, n_strings) con le predizioni filtrate.
    """
    if kernel_size <= 1:
        return predictions

    # kernel_size deve essere dispari per avere un centro simmetrico
    if kernel_size % 2 == 0:
        kernel_size += 1

    def mode_filter(window: np.ndarray) -> float:
        """Restituisce la moda (valore più frequente) della finestra."""
        values, counts = np.unique(window, return_counts=True)
        return values[counts.argmax()]

    filtered = np.empty_like(predictions)
    n_strings = predictions.shape[1]

    for s in range(n_strings):
        filtered[:, s] = generic_filter(
            predictions[:, s].astype(float),
            function=mode_filter,
            size=kernel_size,
            mode="nearest",   # estende il bordo con il valore più vicino
        )

    # Conta quanti frame sono stati modificati (utile per logging)
    changed = int((filtered != predictions).sum())
    total   = predictions.size
    logger.info(
        f"Median filter (k={kernel_size}, ~{kernel_size * config.HOP_LENGTH / config.SAMPLE_RATE * 1000:.0f}ms): "
        f"{changed}/{total} frame modificati ({100*changed/total:.1f}%)"
    )

    return filtered


def _merge_gap_notes(
    notes: list[dict],
    max_gap_s: float = 0.10,
) -> list[dict]:
    """
    Gap Fill: unisce note consecutive con lo stesso pitch e la stessa corda
    che sono separate da un silenzio ≤ max_gap_s secondi.

    Problema che risolve
    ---------------------
    TabCNN predice frame per frame (~11ms cadauno). Quando la confidenza
    del modello scende brevemente sotto soglia (es. per un transitorio
    o un'oscillazione del segnale), il decoder chiude la nota e ne apre
    una nuova al frame successivo. Il risultato è una nota lunga spezzata
    in tanti frammenti corti (chattering).

    Il gap fill opera DOPO il decoder: se due note hanno lo stesso pitch
    e la stessa corda, e il gap tra fine della prima e inizio della seconda
    è ≤ max_gap_s, le fonde in una nota unica estendendo la durata della
    prima fino alla fine della seconda.

    Args:
        notes:      Lista di note già decodificate e filtrate per durata
                    minima. Ogni elemento è un dict con almeno:
                    'time', 'duration', 'pitch', 'string'.
        max_gap_s:  Soglia massima del silenzio (in secondi) affinché due
                    note vengano considerate la stessa nota interrotta.
                    Default 100ms — copre ~9 frame a HOP=512, SR=44100.

    Returns:
        Lista di note con i frammenti uniti. La nota risultante eredita
        tutti i campi della prima, con 'duration' aggiornata all'ampiezza
        totale (dalla fine dell'ultima nota inglobata).
    """
    if not notes:
        return notes

    # Raggruppa le note per (pitch, corda) per confrontarle in parallelo
    # tra corde diverse senza interferenze
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for n in notes:
        key = (n["pitch"], n.get("string", -1))
        groups[key].append(n)

    merged: list[dict] = []
    for key, group in groups.items():
        # Ordina per tempo di onset all'interno del gruppo
        group.sort(key=lambda x: x["time"])

        current = dict(group[0])  # copia difensiva
        for nxt in group[1:]:
            current_end = current["time"] + current["duration"]
            gap = nxt["time"] - current_end

            if gap <= max_gap_s:
                # ── Fusione ────────────────────────────────────────────
                # La nuova durata va dall'onset di 'current' alla fine di 'nxt'
                new_end = nxt["time"] + nxt["duration"]
                current["duration"] = round(new_end - current["time"], 4)
                # Aggiorna la confidenza come media pesata (opzionale)
                if "confidence" in current and "confidence" in nxt:
                    current["confidence"] = round(
                        (current["confidence"] + nxt["confidence"]) / 2, 4
                    )
            else:
                # Gap troppo ampio: chiudi la nota corrente e inizia la prossima
                merged.append(current)
                current = dict(nxt)

        merged.append(current)  # aggiungi l'ultima nota del gruppo

    # Riordina tutte le note per onset (le note di corde diverse si mescolano)
    merged.sort(key=lambda x: (x["time"], x.get("string", -1)))
    return merged


def decode_predictions(
    predictions: np.ndarray,
    confidences: np.ndarray,
    hop_length: int = config.HOP_LENGTH,
    sr: int = config.SAMPLE_RATE,
    confidence_threshold: float = config.ONSET_THRESHOLD,
    gap_fill_s: float = 0.10,
) -> list[dict]:
    """
    Decodifica le predizioni frame-by-frame della TabCNN in una sequenza
    di note discrete.

    Pipeline interna:
      1. Hysteresis (doppia soglia) — riduce il chattering in real-time
      2. Filtro di durata minima    — scarta i click spuri < 50ms
      3. Gap Fill                   — unisce i frammenti della stessa nota
                                      separati da un silenzio ≤ gap_fill_s

    Args:
        predictions:          Matrice (n_frames, num_strings) con il fret
                              predetto per ogni frame e ogni corda.
        confidences:          Matrice (n_frames, num_strings) con la
                              confidenza associata ad ogni predizione.
        hop_length:           Hop length CQT in campioni.
        sr:                   Sample rate in Hz.
        confidence_threshold: Soglia di onset (attivazione nota).
        gap_fill_s:           Gap massimo (secondi) per la fusione note.
                              Imposta a 0.0 per disabilitare il gap fill.
    """
    n_frames, num_strings = predictions.shape
    frame_duration = hop_length / sr  # Durata di un frame in secondi

    notes = []
    prev_fret = [-1] * num_strings  # -1 = corda non attiva
    
    # Soglia di rilascio inferiore per evitare oscillazioni rapide (chattering)
    release_threshold = confidence_threshold * 0.6

    for t in range(n_frames):
        time_sec = t * frame_duration

        for s in range(num_strings):
            fret = int(predictions[t, s])
            conf = float(confidences[t, s])

            # Rimuove valori non validi (20 = non suonata, -1 = vuota)
            if fret < 0 or fret >= config.NUM_FRETS:
                prev_fret[s] = -1
                continue

            # Logica di Hysteresis
            if prev_fret[s] != -1:
                # La corda era attiva nel frame precedente
                if conf < release_threshold:
                    # Rilascio: spegne la nota
                    prev_fret[s] = -1
                else:
                    # La nota continua ad essere attiva
                    if fret == prev_fret[s]:
                        # Aggiorna la durata
                        for note in reversed(notes):
                            if note["string"] == s and note["fret"] == fret:
                                note["duration"] = round(time_sec - note["time"] + frame_duration, 4)
                                break
                    else:
                        # Tasto cambiato: iniziamo una nuova nota se ha abbastanza confidenza
                        if conf >= confidence_threshold:
                            midi_pitch = tab_to_midi_pitch(s, fret)
                            notes.append({
                                "time": round(time_sec, 4),
                                "duration": round(frame_duration, 4),
                                "pitch": midi_pitch,
                                "note_name": midi_to_note_name(midi_pitch),
                                "string": s,
                                "fret": fret,
                                "confidence": round(conf, 4),
                            })
                            prev_fret[s] = fret
                        else:
                            # Se la confidenza è insufficiente per cambiare tasto, spegni
                            prev_fret[s] = -1
            else:
                # La corda non era attiva
                if conf >= confidence_threshold:
                    # Onset: iniziamo una nuova nota
                    midi_pitch = tab_to_midi_pitch(s, fret)
                    notes.append({
                        "time": round(time_sec, 4),
                        "duration": round(frame_duration, 4),
                        "pitch": midi_pitch,
                        "note_name": midi_to_note_name(midi_pitch),
                        "string": s,
                        "fret": fret,
                        "confidence": round(conf, 4),
                    })
                    prev_fret[s] = fret

    # ── Passo 2: filtro durata minima ─────────────────────────────────────
    # Scarta le note brevissime (< 50ms) generate da picchi di rumore
    min_duration = 0.05
    filtered_notes = [n for n in notes if n["duration"] >= min_duration]
    filtered_notes.sort(key=lambda x: (x["time"], x["string"]))

    logger.debug(f"Note dopo filtro durata: {len(filtered_notes)} (erano {len(notes)})")

    # ── Passo 3: Gap Fill ─────────────────────────────────────────────────
    # Unisce i frammenti della stessa nota separati da un silenzio breve.
    # Disabilitabile impostando gap_fill_s=0.0.
    if gap_fill_s > 0.0:
        before = len(filtered_notes)
        filtered_notes = _merge_gap_notes(filtered_notes, max_gap_s=gap_fill_s)
        logger.debug(
            f"Gap fill ({gap_fill_s*1000:.0f}ms): "
            f"{before} → {len(filtered_notes)} note "
            f"(fuse {before - len(filtered_notes)})"
        )

    return filtered_notes


def transcribe_audio(
    audio_path: str,
    model: TabCNN,
    device: str = "cpu",
) -> list[dict]:
    """
    Pipeline completa di trascrizione con amt_tools:
    1. Estrazione CQT
    2. Modello pre_proc, forward
    3. Estrazione confidenze reali dalla distribuzione softmax
    4. Modello post_proc (argmax → fret predetti)
    5. Decodifica frame-wise → sequenza di note
    """
    logger.info(f"Trascrizione audio: {audio_path}")
    from amt_tools import tools

    # 1. Preprocessing: estrazione CQT grezza
    input_tensor, sr, n_frames_orig = prepare_input_tensor(audio_path, device)
    batch = {tools.KEY_FEATS: input_tensor}

    # 2. Inferenza modello amt_tools
    with torch.no_grad():
        batch = model.pre_proc(batch)
        output = model(batch[tools.KEY_FEATS])
        batch[tools.KEY_OUTPUT] = output

        # ── Estrazione confidenze REALI dalla distribuzione softmax ────────
        #
        # output[KEY_TABLATURE] contiene la distribuzione softmax prodotta da
        # SoftmaxGroups PRIMA che post_proc() la converta in argmax.
        # È l'unica finestra in cui le probabilità su tutte le 21 classi
        # (20 fret + silenzio) sono ancora disponibili.
        #
        # Per ogni (frame t, corda s) la confidenza è:
        #   conf[t, s] = max_j ( softmax[t, s, j] )  con j ∈ [0..20]
        #
        # Se il modello è sicuro:  [0.01, ..., 0.92, ..., 0.01] → conf = 0.92
        # Se il modello è incerto: [0.06, ..., 0.11, ..., 0.08] → conf = 0.11
        #
        # Questa informazione viene poi usata dall'hysteresis in
        # decode_predictions() per decidere quando aprire/chiudere una nota.
        raw_softmax = output[tools.KEY_TABLATURE]  # (1, T, n_strings, n_classes)

        try:
            raw_np = (
                raw_softmax.cpu().numpy()
                if isinstance(raw_softmax, torch.Tensor)
                else np.array(raw_softmax)
            )

            logger.info(f"raw_softmax shape originale: {raw_np.shape}")

            # Rimuovi tutte le dimensioni unitarie e la dimensione batch
            raw_np = np.squeeze(raw_np)   # (1, T, 126) → (T, 126)

            # SoftmaxGroups appiattisce: n_strings × n_classes = 6 × 21 = 126
            n_strings = config.NUM_STRINGS   # 6
            n_classes = config.NUM_CLASSES   # 21  (20 fret + silenzio)

            if raw_np.ndim == 2:
                T, last_dim = raw_np.shape
                if last_dim == n_strings * n_classes:
                    raw_np = raw_np.reshape(T, n_strings, n_classes)
                elif last_dim != n_strings:
                    raise ValueError(
                        f"Shape (T={T}, last_dim={last_dim}) non gestita. "
                        f"Atteso {n_strings * n_classes} o {n_strings}."
                    )
            # raw_np è ora (T, 6, 21) o (T, 6)

            if raw_np.ndim == 3:
                # I valori sono log-prob (tutti ≤ 0).
                # exp() causa underflow numerico per log-prob << 0 (es. -307).
                #
                # Soluzione: usiamo il MAX dei log-prob per corda come
                # indicatore di confidenza — è l'unico valore < 0 ma vicino
                # a 0, tutto il resto è molto più negativo.
                #
                # Per convertirlo in una confidenza in [0, 1] normalizziamo
                # con un min-max per corda su tutti i frame:
                #   conf_norm[t, s] = (log_max[t,s] - min_s) / (max_s - min_s)
                #
                # Dove max_s ≈ 0 (la classe più probabile) e min_s è il min
                # globale (classe meno probabile).
                log_max = raw_np.max(axis=-1)  # (T, 6)  — valori in (-inf, 0]

                # Normalizzazione per corda: porta ogni corda in [0, 1]
                s_min = log_max.min(axis=0, keepdims=True)   # (1, 6)
                s_max = log_max.max(axis=0, keepdims=True)   # (1, 6)
                denom = s_max - s_min
                denom[denom == 0] = 1.0                       # evita div/0
                conf_np = (log_max - s_min) / denom           # ∈ [0, 1]
            else:
                # (T, 6) — già un proxy di confidenza
                conf_np = raw_np

            logger.info(
                f"Confidenze reali estratte — shape: {conf_np.shape} | "
                f"media: {conf_np.mean():.3f} | "
                f"min: {conf_np.min():.3f} | "
                f"max: {conf_np.max():.3f}"
            )

        except Exception as e:
            # Fallback sicuro: se l'estrazione fallisce (API amt_tools cambiata,
            # shape inattesa, ecc.) usiamo 1.0 e avvisiamo nei log.
            logger.warning(
                f"Impossibile estrarre le confidenze reali ({e}). "
                "Fallback a confidenze=1.0 — l'hysteresis sarà disabilitata."
            )
            conf_np = None

        # post_proc converte la distribuzione softmax in argmax (fret predetto)
        risultato_finale = model.post_proc(batch)

    # 3. Estrazione matrice fret per frame: porta a forma (T, 6)
    tablatura_stimata = risultato_finale[tools.KEY_TABLATURE]

    if isinstance(tablatura_stimata, torch.Tensor):
        tab_np = tablatura_stimata.cpu().numpy()
    else:
        tab_np = tablatura_stimata

    tab_np = np.squeeze(tab_np)
    if tab_np.ndim == 1:
        tab_np = tab_np.reshape(-1, 6)
    elif tab_np.ndim == 3 and tab_np.shape[1] == 6:
        tab_np = tab_np[:, :, 0]

    predictions = tab_np

    # 4. Scegli la matrice di confidenze da passare al decoder
    if conf_np is not None and conf_np.shape == predictions.shape:
        confidences = conf_np
        logger.info("Hysteresis attiva con confidenze reali del modello.")
    else:
        # Shape non coincide (raro) o estrazione fallita: fallback sicuro
        confidences = np.ones_like(predictions)
        logger.warning(
            f"Shape confidenze {getattr(conf_np, 'shape', None)} != "
            f"predizioni {predictions.shape}. Fallback a 1.0."
        )

    logger.info(f"Shape predizioni: {predictions.shape}")

    # 5. Median Filter: smooting prima del decoder
    #    Sostituisce ogni frame con la moda della finestra temporale → elimina
    #    i dropout brevi che il decoder interpreterebbe come fine nota.
    predictions = _median_filter_predictions(predictions, kernel_size=9)

    # 6. Decodifica: predizioni frame-wise → sequenza di note
    notes = decode_predictions(predictions, confidences)
    logger.info(f"Note trascritte: {len(notes)}")

    return notes

