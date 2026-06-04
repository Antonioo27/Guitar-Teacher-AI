"""
dataset.py — Modulo 1: Parsing delle annotazioni e gestione del dataset.

Gestisce il parsing dei dati di ground truth da diversi formati:
- JAMS (usato da GuitarSet): annotazioni precise con onset, pitch e durata.
- MIDI: spartiti musicali in formato standard.

Fornisce inoltre un Dataset PyTorch per training/evaluation (usato nel
notebook Colab, non nella pipeline di inferenza).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import jams
import pretty_midi

from . import config


# =============================================================================
# Utilità di conversione note
# =============================================================================

def midi_to_note_name(midi_pitch: int) -> str:
    """
    Converte un numero MIDI pitch nel nome della nota corrispondente.

    Args:
        midi_pitch: Numero MIDI (0-127). Es: 60 = C4, 69 = A4.

    Returns:
        Nome della nota con ottava. Es: "C4", "A#3".
    """
    if midi_pitch < 0 or midi_pitch > 127:
        return "?"
    note_name = config.NOTE_NAMES[midi_pitch % 12]
    octave = (midi_pitch // 12) - 1
    return f"{note_name}{octave}"


def note_name_to_midi(note_name: str) -> int:
    """
    Converte il nome di una nota nel corrispondente MIDI pitch.

    Args:
        note_name: Nome della nota (es. "C4", "A#3").

    Returns:
        MIDI pitch (0-127).
    """
    # Separa nome nota e ottava
    if "#" in note_name:
        note = note_name[:2]
        octave = int(note_name[2:])
    else:
        note = note_name[0]
        octave = int(note_name[1:])

    pitch_class = config.NOTE_NAMES.index(note)
    return pitch_class + (octave + 1) * 12


def tab_to_midi_pitch(string_idx: int, fret: int) -> int:
    """
    Converte una posizione sulla tastiera della chitarra in MIDI pitch.

    Args:
        string_idx: Indice della corda (0-5, dove 0 = E2 bassa).
        fret: Numero del tasto (0 = corda a vuoto).

    Returns:
        MIDI pitch corrispondente.
    """
    return config.GUITAR_TUNING[string_idx] + fret


# =============================================================================
# Parsing JAMS (GuitarSet)
# =============================================================================

def parse_jams(jams_path: str) -> list[dict[str, Any]]:
    """
    Legge un file JAMS (formato usato da GuitarSet) e ne estrae le
    annotazioni nota per nota.

    GuitarSet fornisce annotazioni per-corda con onset/offset e pitch
    come valori MIDI continui (con micro-intonazione). Qui arrotondiamo
    al semitono più vicino.

    Args:
        jams_path: Percorso al file .jams.

    Returns:
        Lista di dizionari con campi:
        - "time": float — tempo di onset in secondi
        - "duration": float — durata in secondi
        - "midi_pitch": int — pitch MIDI (arrotondato al semitono)
        - "note_name": str — nome della nota (es. "E4")
        - "string": int — indice della corda (0-5) se disponibile
        - "fret": int — tasto premuto se calcolabile
    """
    jam = jams.load(jams_path)
    annotations = []

    # GuitarSet usa il namespace "note_midi" o "pitch_midi" per le annotazioni
    for i, ann in enumerate(jam.annotations):
        if ann.namespace not in ("note_midi", "pitch_midi", "note_hz", "pitch_contour"):
            continue

        for obs in ann.data:
            time = obs.time
            duration = obs.duration

            # Il valore può essere pitch MIDI (float) o frequenza Hz
            if ann.namespace in ("note_midi", "pitch_midi"):
                midi_pitch = int(round(obs.value))
            elif ann.namespace == "note_hz":
                # Conversione Hz → MIDI
                midi_pitch = int(round(librosa.hz_to_midi(obs.value)))
            else:
                continue

            # Calcola il tasto se possibile (corda nota dall'indice dell'annotazione)
            string_idx = min(i, config.NUM_STRINGS - 1)
            open_string_pitch = config.GUITAR_TUNING[string_idx]
            fret = midi_pitch - open_string_pitch

            annotations.append({
                "time": float(time),
                "duration": float(duration),
                "midi_pitch": midi_pitch,
                "note_name": midi_to_note_name(midi_pitch),
                "string": string_idx,
                "fret": max(0, fret),
            })

    # Ordina per tempo di onset
    annotations.sort(key=lambda x: x["time"])
    return annotations


def parse_midi(midi_path: str) -> list[dict[str, Any]]:
    """
    Legge uno spartito MIDI e ne estrae le note come sequenza temporale.

    Usato per caricare lo spartito "ideale" dell'esercizio da confrontare
    con la trascrizione dell'IA nel Modulo 3 (DTW).

    Args:
        midi_path: Percorso al file .mid/.midi.

    Returns:
        Lista di dizionari con campi:
        - "time": float — tempo di onset in secondi
        - "duration": float — durata in secondi
        - "midi_pitch": int — pitch MIDI
        - "note_name": str — nome della nota
        - "velocity": int — velocità/dinamica MIDI
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    annotations = []

    for instrument in midi_data.instruments:
        # Ignora canali di percussione
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            annotations.append({
                "time": float(note.start),
                "duration": float(note.end - note.start),
                "midi_pitch": note.pitch,
                "note_name": midi_to_note_name(note.pitch),
                "velocity": note.velocity,
            })

    # Ordina per tempo di onset
    annotations.sort(key=lambda x: x["time"])
    return annotations


# =============================================================================
# Costruzione sequenze per il DTW
# =============================================================================

def build_note_sequence(
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Converte una lista di annotazioni in una sequenza semplificata
    di note, pronta per l'allineamento DTW.

    Rimuove informazioni non necessarie e mantiene solo time, pitch e nome.

    Args:
        annotations: Lista di annotazioni da parse_jams() o parse_midi().

    Returns:
        Lista di {"time": float, "pitch": int, "note_name": str}.
    """
    return [
        {
            "time": ann["time"],
            "pitch": ann["midi_pitch"],
            "note_name": ann["note_name"],
        }
        for ann in annotations
    ]


# =============================================================================
# Dataset PyTorch per GuitarSet (Training/Evaluation)
# =============================================================================

class GuitarSetDataset:
    """
    Dataset per il caricamento dei dati GuitarSet.
    Usato principalmente nel notebook Colab per training e evaluation.
    Per la pipeline di inferenza, usare direttamente le funzioni
    di audio_processing.py.

    Attributes:
        data_dir: Directory radice di GuitarSet.
        audio_files: Lista dei percorsi ai file audio.
        jams_files: Lista dei percorsi ai file JAMS corrispondenti.
    """

    def __init__(
        self,
        data_dir: str | Path = config.GUITARSET_DIR,
        audio_subdir: str = "audio_mono-mic",
        jams_subdir: str = "annotation",
    ):
        self.data_dir = Path(data_dir)
        self.audio_dir = self.data_dir / audio_subdir
        self.jams_dir = self.data_dir / jams_subdir

        # Trova tutte le coppie audio-annotazione
        self.audio_files = sorted(self.audio_dir.glob("*.wav")) if self.audio_dir.exists() else []
        self.jams_files = sorted(self.jams_dir.glob("*.jams")) if self.jams_dir.exists() else []

    def __len__(self) -> int:
        return len(self.audio_files)

    def get_pair(self, idx: int) -> tuple[str, str]:
        """Restituisce la coppia (audio_path, jams_path) all'indice dato."""
        return str(self.audio_files[idx]), str(self.jams_files[idx])

    def get_all_annotations(self) -> list[dict[str, Any]]:
        """Carica tutte le annotazioni JAMS del dataset."""
        all_annotations = []
        for jams_path in self.jams_files:
            annotations = parse_jams(str(jams_path))
            all_annotations.extend(annotations)
        return all_annotations
