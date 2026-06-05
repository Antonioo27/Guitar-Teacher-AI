"""
midi_inspector.py — Strumento di ispezione e debug per file MIDI.

Fornisce due output a partire da un file .mid:
  1. Testuale/Strutturato: un pandas.DataFrame con le note estratte,
     pronto per essere passato all'algoritmo DTW (Modulo 3).
  2. Visivo: un Piano Roll con matplotlib per il debugging umano.

Uso standalone:
    python -m guitar_tutor_pipeline.src.midi_inspector

Uso come libreria:
    from guitar_tutor_pipeline.src.midi_inspector import MidiInspector

    inspector = MidiInspector("esercizio.mid")
    df = inspector.to_dataframe()
    inspector.plot_piano_roll()
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import pretty_midi


# =============================================================================
# Classe principale
# =============================================================================

class MidiInspector:
    """
    Incapsula la logica di lettura, estrazione dati e visualizzazione
    di un file MIDI.

    Attributes:
        midi_path (Path): Percorso al file .mid analizzato.
        midi_data (pretty_midi.PrettyMIDI): Oggetto MIDI parsato.
    """

    def __init__(self, midi_path: str | Path):
        """
        Carica e parsifica il file MIDI.

        Args:
            midi_path: Percorso al file .mid o .midi.

        Raises:
            FileNotFoundError: Se il file non esiste.
            Exception: Se pretty_midi non riesce a leggere il file.
        """
        self.midi_path = Path(midi_path)
        if not self.midi_path.exists():
            raise FileNotFoundError(f"File MIDI non trovato: {self.midi_path}")

        self.midi_data = pretty_midi.PrettyMIDI(str(self.midi_path))

    # -------------------------------------------------------------------------
    # Componente 1 — Estrazione dati strutturati (DataFrame)
    # -------------------------------------------------------------------------

    def to_dataframe(self, include_drums: bool = False) -> pd.DataFrame:
        """
        Estrae tutte le note del file MIDI in un pandas.DataFrame.

        Ogni riga corrisponde a una nota suonata, ordinata per tempo di onset.
        Il DataFrame è pronto per essere normalizzato e passato al DTW.

        Args:
            include_drums: Se True, include anche le note delle tracce
                           percussive (canale 10). Default False.

        Returns:
            pd.DataFrame con colonne:
                - start_time  (float): Tempo di inizio in secondi.
                - end_time    (float): Tempo di fine in secondi.
                - duration    (float): Durata in secondi (end - start).
                - pitch       (int):   Numero MIDI della nota (0-127).
                - pitch_name  (str):   Nome della nota, es. "C4", "A#3".
                - velocity    (int):   Velocità/dinamica MIDI (0-127).
                - instrument  (str):   Nome dello strumento della traccia.
                - track_idx   (int):   Indice della traccia di provenienza.
        """
        rows: list[dict] = []

        for track_idx, instrument in enumerate(self.midi_data.instruments):
            # Salta le tracce percussive se richiesto
            if instrument.is_drum and not include_drums:
                continue

            for note in instrument.notes:
                rows.append({
                    "start_time": round(float(note.start), 6),
                    "end_time":   round(float(note.end), 6),
                    "duration":   round(float(note.end - note.start), 6),
                    "pitch":      int(note.pitch),
                    # pretty_midi restituisce es. "C4", "A#3"
                    "pitch_name": pretty_midi.note_number_to_name(note.pitch),
                    "velocity":   int(note.velocity),
                    "instrument": instrument.name or f"Traccia {track_idx}",
                    "track_idx":  track_idx,
                })

        if not rows:
            # Restituisce DataFrame vuoto ma con schema corretto
            return pd.DataFrame(columns=[
                "start_time", "end_time", "duration",
                "pitch", "pitch_name", "velocity",
                "instrument", "track_idx",
            ])

        df = pd.DataFrame(rows)
        # Ordina per onset e poi per pitch (note simultanee)
        df.sort_values(["start_time", "pitch"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # -------------------------------------------------------------------------
    # Componente 2 — Visualizzazione Piano Roll
    # -------------------------------------------------------------------------

    def plot_piano_roll(
        self,
        figsize: tuple[float, float] = (14, 6),
        bar_height: float = 0.7,
        colormap: str = "tab10",
        title: Optional[str] = None,
        show: bool = True,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """
        Genera un Piano Roll del file MIDI con matplotlib.

        Ogni nota è rappresentata come un segmento orizzontale colorato:
          - Asse X = tempo (secondi)
          - Asse Y = pitch MIDI (numero della nota)
          - Colore = traccia/strumento di provenienza

        Args:
            figsize:    Dimensioni della figura (larghezza, altezza) in pollici.
            bar_height: Altezza di ciascuna barra (0 < bar_height <= 1).
            colormap:   Colormap matplotlib per distinguere le tracce.
            title:      Titolo personalizzato del grafico. Se None, usa il
                        nome del file.
            show:       Se True, chiama plt.show() alla fine.
            save_path:  Se fornito, salva la figura nel percorso specificato.

        Returns:
            L'oggetto matplotlib.figure.Figure generato.
        """
        df = self.to_dataframe()

        if df.empty:
            print("[MidiInspector] Nessuna nota trovata: il Piano Roll è vuoto.")
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title("Piano Roll — nessuna nota")
            if show:
                plt.show()
            return fig

        # --- Preparazione colori per traccia ---
        tracks = df["track_idx"].unique()
        cmap = plt.get_cmap(colormap)
        color_map = {t: cmap(i % 10) for i, t in enumerate(tracks)}

        # --- Calcolo limiti assi ---
        x_min = df["start_time"].min()
        x_max = df["end_time"].max()
        y_min = df["pitch"].min() - 2
        y_max = df["pitch"].max() + 2

        fig, ax = plt.subplots(figsize=figsize)

        # --- Disegno delle note come barre orizzontali ---
        for _, row in df.iterrows():
            ax.broken_barh(
                [(row["start_time"], row["duration"])],   # (xmin, xwidth)
                (row["pitch"] - bar_height / 2, bar_height),  # (ymin, yheight)
                facecolors=color_map[row["track_idx"]],
                edgecolors="white",
                linewidth=0.3,
                alpha=0.85,
            )

        # --- Etichette asse Y: mostra il nome della nota ogni 12 semitoni ---
        y_ticks = range(
            max(0, int(y_min)),
            min(128, int(y_max) + 1),
            12,
        )
        y_labels = [
            f"{pretty_midi.note_number_to_name(p)}  ({p})" for p in y_ticks
        ]
        ax.set_yticks(list(y_ticks))
        ax.set_yticklabels(y_labels, fontsize=8)

        # --- Griglia leggera sulle ottave ---
        for tick in y_ticks:
            ax.axhline(tick, color="gray", linewidth=0.4, linestyle="--", alpha=0.4)

        # --- Legenda per tracce ---
        legend_handles = [
            mpatches.Patch(
                color=color_map[t],
                label=df.loc[df["track_idx"] == t, "instrument"].iloc[0],
            )
            for t in tracks
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

        # --- Configurazione assi e titolo ---
        ax.set_xlim(x_min - 0.1, x_max + 0.1)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Tempo (secondi)", fontsize=10)
        ax.set_ylabel("Pitch MIDI", fontsize=10)
        ax.set_title(
            title or f"Piano Roll — {self.midi_path.name}",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(axis="x", color="gray", linewidth=0.3, alpha=0.4)
        fig.tight_layout()

        if save_path:
            fig.savefig(str(save_path), dpi=150)
            print(f"[MidiInspector] Piano Roll salvato in: {save_path}")

        if show:
            plt.show()

        return fig

    # -------------------------------------------------------------------------
    # Metodi di utilità
    # -------------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Restituisce un dizionario con le informazioni generali del file MIDI.

        Returns:
            dict con: durata totale, numero di tracce, numero totale di note,
            tempo stimato (BPM), e risoluzione (tick per beat).
        """
        df = self.to_dataframe()
        tempos = self.midi_data.get_tempo_changes()
        avg_bpm = float(tempos[1].mean()) if len(tempos[1]) > 0 else 120.0

        return {
            "file":           self.midi_path.name,
            "duration_s":     round(self.midi_data.get_end_time(), 3),
            "num_tracks":     len(self.midi_data.instruments),
            "total_notes":    len(df),
            "avg_bpm":        round(avg_bpm, 1),
            "resolution_ppq": self.midi_data.resolution,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"MidiInspector({s['file']} | "
            f"{s['duration_s']}s | "
            f"{s['num_tracks']} tracce | "
            f"{s['total_notes']} note | "
            f"~{s['avg_bpm']} BPM)"
        )


# =============================================================================
# Demo autonoma — crea un MIDI fittizio e testa entrambi gli output
# =============================================================================

if __name__ == "__main__":
    import os

    print("=" * 60)
    print("  MidiInspector — Demo autonoma")
    print("=" * 60)

    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    guitar = pretty_midi.Instrument(program=25, name="Acoustic Guitar")

    note_definitions = [
        (52, 0.00, 0.45, 80),
        (57, 0.50, 0.95, 82),
        (62, 1.00, 1.45, 78),
        (67, 1.50, 1.95, 85),
        (71, 2.00, 2.60, 75),
    ]

    for pitch, start, end, vel in note_definitions:
        guitar.notes.append(
            pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end)
        )

    pm.instruments.append(guitar)

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name
    pm.write(tmp_path)

    inspector = MidiInspector(tmp_path)
    print(f"\n[Inspector] {inspector}\n")
    print(inspector.to_dataframe().to_string())
    inspector.plot_piano_roll(title="Demo", show=False)
    os.unlink(tmp_path)
    print("\n[+] Demo completata.")
