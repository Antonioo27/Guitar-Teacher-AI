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


def suppress_harmonics(probs: np.ndarray, threshold: float = 0.5, suppression_factor: float = 0.1) -> np.ndarray:
    """
    Sopprime le armoniche spurie (+12 e +19 semitoni) se c'è una fondamentale forte.
    Operiamo direttamente sul tensore delle probabilità (T, 6, 21).
    """
    T, num_strings, num_classes = probs.shape
    num_frets = 20 # 0..19
    tuning = [40, 45, 50, 55, 59, 64] # E2, A2, D3, G3, B3, E4
    
    # Crea una matrice di mapping (string, fret) -> midi_pitch
    pitch_map = np.zeros((num_strings, num_frets), dtype=int)
    for s in range(num_strings):
        for f in range(num_frets):
            pitch_map[s, f] = tuning[s] + f
            
    for t in range(T):
        active_pitches = []
        for s in range(num_strings):
            for f in range(num_frets):
                if probs[t, s, f] >= threshold:
                    active_pitches.append(pitch_map[s, f])
                    
        if active_pitches:
            for s in range(num_strings):
                for f in range(num_frets):
                    p = pitch_map[s, f]
                    for ap in active_pitches:
                        if p == ap + 12 or p == ap + 19:
                            probs[t, s, f] *= suppression_factor
                            break
    return probs


def _filter_implausible_notes(
    notes: list[dict],
    min_duration_s: float = 0.08,
) -> list[dict]:
    """
    Filtro di Plausibilità: rimuove note fisicamente impossibili o spurie.

    Criteri applicati
    -----------------
    1. **Durata minima**: note < 80ms sono quasi sempre artefatti del modello
       (il minimo musicalmente significativo a 120 BPM è una semicroma = 125ms).
       Una soglia a 80ms preserva le semicrome ma elimina i click di 1-7 frame.

    2. **Range MIDI valido**: la chitarra in accordatura standard suona tra
       E2 (MIDI 40, corda VI open) e l'ultimo fret della corda I.
       Note fuori da questa finestra sono artefatti (es. C3 in contesti
       dove la nota più bassa è A2).

    Args:
        notes:          Lista di note prodotta da decode_predictions().
        min_duration_s: Durata minima in secondi. Default 80ms.

    Returns:
        Lista filtrata con solo note plausibili.
    """
    # Range MIDI chitarra standard E2=40 … (E4 + 20 fret) = 84
    min_pitch = config.GUITAR_TUNING[0]                          # E2 = 40
    max_pitch = config.GUITAR_TUNING[-1] + config.NUM_FRETS      # E4 + 20 = 84

    plausible = []
    removed   = []

    for n in notes:
        reason = None
        if n["duration"] < min_duration_s:
            reason = f"durata {n['duration']*1000:.0f}ms < {min_duration_s*1000:.0f}ms"
        elif not (min_pitch <= n["pitch"] <= max_pitch):
            reason = f"pitch {n['pitch']} fuori range [{min_pitch},{max_pitch}]"

        if reason:
            removed.append((n["note_name"], round(n["time"], 2), reason))
        else:
            plausible.append(n)

    if removed:
        for name, t, why in removed:
            logger.debug(f"  Rimossa nota spuria: {name} @{t}s — {why}")
        logger.info(
            f"Filtro plausibilità: rimoss{'a' if len(removed)==1 else 'e'} "
            f"{len(removed)} not{'a' if len(removed)==1 else 'e'} spur{'ia' if len(removed)==1 else 'ie'} "
            f"({', '.join(r[0] for r in removed)})"
        )

    return plausible


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
    probs: np.ndarray,
    hop_length: int = config.HOP_LENGTH,
    sr: int = config.SAMPLE_RATE,
    t_on: float = 0.50,
    t_off: float = 0.25,
    gap_fill_s: float = 0.0,
) -> list[dict]:
    """
    Decodifica la matrice delle probabilità usando Hysteresis a doppia soglia (Schmitt Trigger).
    """
    n_frames, num_strings, _ = probs.shape
    frame_duration = hop_length / sr
    notes = []
    
    # -1 significa corda inattiva (silenzio)
    active_fret = [-1] * num_strings
    
    for t in range(n_frames):
        time_sec = t * frame_duration
        
        for s in range(num_strings):
            current_fret = active_fret[s]
            
            if current_fret == -1:
                best_fret = int(np.argmax(probs[t, s, :20]))
                if probs[t, s, best_fret] >= t_on:
                    active_fret[s] = best_fret
                    midi_pitch = tab_to_midi_pitch(s, best_fret)
                    notes.append({
                        "time": round(time_sec, 4),
                        "duration": round(frame_duration, 4),
                        "pitch": midi_pitch,
                        "note_name": midi_to_note_name(midi_pitch),
                        "string": s,
                        "fret": best_fret,
                        "confidence": round(float(probs[t, s, best_fret]), 4),
                    })
            else:
                if probs[t, s, current_fret] < t_off:
                    best_other_fret = int(np.argmax(probs[t, s, :20]))
                    if probs[t, s, best_other_fret] >= t_on:
                        active_fret[s] = best_other_fret
                        midi_pitch = tab_to_midi_pitch(s, best_other_fret)
                        notes.append({
                            "time": round(time_sec, 4),
                            "duration": round(frame_duration, 4),
                            "pitch": midi_pitch,
                            "note_name": midi_to_note_name(midi_pitch),
                            "string": s,
                            "fret": best_other_fret,
                            "confidence": round(float(probs[t, s, best_other_fret]), 4),
                        })
                    else:
                        active_fret[s] = -1 
                else:
                    for note in reversed(notes):
                        if note["string"] == s and note["fret"] == current_fret:
                            note["duration"] = round(time_sec - note["time"] + frame_duration, 4)
                            note["confidence"] = round(max(note["confidence"], float(probs[t, s, current_fret])), 4)
                            break

    # ── Passo 2: filtro durata minima ─────────────────────────────────────
    min_duration = 0.05
    filtered_notes = [n for n in notes if n["duration"] >= min_duration]
    filtered_notes.sort(key=lambda x: (x["time"], x["string"]))

    # ── Passo 3: Gap Fill ─────────────────────────────────────────────────
    if gap_fill_s > 0.0:
        filtered_notes = _merge_gap_notes(filtered_notes, max_gap_s=gap_fill_s)

    # ── Passo 4: Filtro di plausibilità ───────────────────────────────────
    before = len(filtered_notes)
    filtered_notes = _filter_implausible_notes(filtered_notes, min_duration_s=0.05)
    logger.debug(f"Filtro plausibilità: {before} → {len(filtered_notes)} note")

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
        logger.info(f"[DEBUG] raw_softmax type: {type(output[tools.KEY_TABLATURE])}")
        if hasattr(output[tools.KEY_TABLATURE], 'shape'):
            logger.info(f"[DEBUG] raw_softmax shape: {output[tools.KEY_TABLATURE].shape}")
        elif isinstance(output[tools.KEY_TABLATURE], list):
            logger.info(f"[DEBUG] raw_softmax list len: {len(output[tools.KEY_TABLATURE])}")
        
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
        final_probs = None

        try:
            raw_np = (
                raw_softmax.cpu().numpy()
                if isinstance(raw_softmax, torch.Tensor)
                else np.array(raw_softmax)
            )

            # Squeeze eventuale dimensione batch se = 1. E.g. (1, T, 126) -> (T, 126)
            if raw_np.ndim == 3 and raw_np.shape[0] == 1:
                raw_np = np.squeeze(raw_np, axis=0)
            
            # Se è (T, 1, 126), rimuovi la dimensione 1
            if raw_np.ndim == 3 and raw_np.shape[1] == 1 and raw_np.shape[2] == 126:
                raw_np = np.squeeze(raw_np, axis=1)

            if raw_np.ndim == 2 and raw_np.shape[1] == 126:
                T = raw_np.shape[0]
                raw_np = raw_np.reshape(T, 6, 21)
                
            import scipy.ndimage
            
            if raw_np.ndim == 3 and raw_np.shape[1] == 6 and raw_np.shape[2] == 21:
                exp_logits = np.exp(raw_np - np.max(raw_np, axis=-1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                probs = scipy.ndimage.gaussian_filter1d(probs, sigma=3.0, axis=0)
                probs = suppress_harmonics(probs, threshold=0.5, suppression_factor=0.1)
                final_probs = probs
                
                smoothed_logits = np.log(probs + 1e-9)
                batch[tools.KEY_OUTPUT][tools.KEY_TABLATURE] = torch.tensor(
                    smoothed_logits.reshape(raw_softmax.shape),
                    device=raw_softmax.device,
                    dtype=raw_softmax.dtype
                )
                conf_np = probs.max(axis=-1)
            else:
                raise ValueError(f"Shape inattesa per raw_softmax: {raw_np.shape}")

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

    # tab_np shape could be (6, T) or (T, 6)
    if tab_np.shape[0] == 6 and tab_np.ndim == 2:
        tab_np = tab_np.T

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

    if final_probs is not None:
        notes = decode_predictions(final_probs)
    else:
        notes = []
    logger.info(f"Note trascritte: {len(notes)}")

    return notes

