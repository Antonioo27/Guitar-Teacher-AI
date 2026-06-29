"""
compareWithMidi.py — Visualizza le note predette da TabCNN accanto allo spartito MIDI.

Scopo (semplice): prendere ciò che produce runTabCNN e metterlo a confronto
con lo spartito di riferimento (.mid). Il DTW serve SOLO a posizionare al
meglio le due sequenze, accoppiando ogni nota predetta con quella del MIDI
a cui corrisponde — anche se le due hanno scale temporali diverse.

Visualizzazione: piano roll a due corsie
    - in alto:  ciò che è stato SUONATO (predizione TabCNN)
    - in basso: lo SPARTITO (riferimento MIDI)
    - linee di collegamento = accoppiamento trovato dal DTW
        verde = stesso pitch, rosso = pitch diverso

Uso:
    # 1) genera le note con runTabCNN
    python -m guitar_tutor_pipeline.src.runTabCNN matrice.npy --output_dir output_notes/cmaj
    # 2) confronta col MIDI
    python -m guitar_tutor_pipeline.src.compareWithMidi \\
        output_notes/cmaj/00_CMaj_120_cqt_notes.json \\
        SetTestHomeMade/00_CMaj_120.mid \\
        --output_dir output_notes/cmaj
"""

import sys
import json
import logging
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, ConnectionPatch

from ..app.dataset import parse_midi, midi_to_note_name
from ..app.alignment import compute_dtw_alignment, pitch_only_distance
from .metrics import compute_all_metrics, print_metrics, save_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PRED_COLOR = "#1f77b4"   # blu — note suonate
REF_COLOR = "#7f7f7f"    # grigio — spartito
MATCH_COLOR = "#2ca02c"  # verde — collegamento DTW con pitch uguale
MISS_COLOR = "#d62728"   # rosso — collegamento DTW con pitch diverso


def _pitch(n: dict) -> int:
    """Pitch MIDI della nota, indipendente dalla sorgente (pitch/midi_pitch)."""
    return n.get("pitch", n.get("midi_pitch"))


# =========================================================================
# Caricamento note predette (JSON di runTabCNN)
# =========================================================================
def load_predicted_notes(json_path: str) -> list[dict]:
    path = Path(json_path)
    if not path.exists():
        logger.error(f"File note predette non trovato: {json_path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        notes = json.load(f)
    logger.info(f"  Note predette caricate: {len(notes)} da {path.name}")
    return notes


# =========================================================================
# Disegno di una corsia (piano roll) su un asse
# =========================================================================
def _draw_lane(ax, notes: list[dict], color: str, title: str) -> None:
    bar_h = 0.8
    for n in notes:
        ax.broken_barh(
            [(n["time"], max(n.get("duration", 0.05), 0.02))],
            (_pitch(n) - bar_h / 2, bar_h),
            facecolors=color,
            edgecolors="white",
            linewidths=0.5,
        )
    ax.set_ylabel(title)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    pitches = [_pitch(n) for n in notes] or [60]
    lo, hi = min(pitches) - 2, max(pitches) + 2
    ax.set_ylim(lo, hi)
    yticks = list(range(lo, hi + 1))
    ax.set_yticks(yticks)
    ax.set_yticklabels([midi_to_note_name(p) for p in yticks], fontsize=8)


# =========================================================================
# Piano roll a due corsie con collegamenti DTW
# =========================================================================
def plot_comparison(
    predicted: list[dict],
    reference: list[dict],
    path: list[tuple[int, int]],
    midi_path: str,
    output_path: Path,
) -> None:
    fig, (ax_pred, ax_ref) = plt.subplots(
        2, 1, figsize=(14, 7), gridspec_kw={"hspace": 0.25}
    )

    _draw_lane(ax_pred, predicted, PRED_COLOR, "SUONATO\n(TabCNN)")
    _draw_lane(ax_ref, reference, REF_COLOR, "SPARTITO\n(MIDI)")

    ax_ref.set_xlabel("Tempo (s)")
    ax_pred.set_title(f"Confronto TabCNN vs spartito — {Path(midi_path).name}")

    # Linee di collegamento DTW: dal centro della nota predetta a quella di rif.
    n_match = 0
    for p_idx, r_idx in path:
        if p_idx >= len(predicted) or r_idx >= len(reference):
            continue
        p, r = predicted[p_idx], reference[r_idx]
        same_pitch = _pitch(p) == _pitch(r)
        n_match += int(same_pitch)

        x_p = p["time"] + max(p.get("duration", 0.05), 0.02) / 2
        x_r = r["time"] + max(r.get("duration", 0.05), 0.02) / 2
        con = ConnectionPatch(
            xyA=(x_p, _pitch(p)), coordsA=ax_pred.transData,
            xyB=(x_r, _pitch(r)), coordsB=ax_ref.transData,
            color=MATCH_COLOR if same_pitch else MISS_COLOR,
            alpha=0.5, linewidth=1.2, zorder=0,
        )
        fig.add_artist(con)

    # Legenda
    legend_elems = [
        Patch(facecolor=PRED_COLOR, label="suonato (TabCNN)"),
        Patch(facecolor=REF_COLOR, label="spartito (MIDI)"),
        plt.Line2D([0], [0], color=MATCH_COLOR, label="DTW: pitch uguale"),
        plt.Line2D([0], [0], color=MISS_COLOR, label="DTW: pitch diverso"),
    ]
    ax_pred.legend(handles=legend_elems, loc="upper right", fontsize=8, ncol=2)

    fig.suptitle(
        f"DTW: {len(path)} coppie  |  pitch coincidenti: {n_match}/{len(path)}",
        y=0.02, fontsize=9,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Piano roll salvato: {output_path}")


# =========================================================================
# Vista sovrapposta (stesso asse) per leggere lo sfasamento temporale
# =========================================================================
def plot_overlay(
    predicted: list[dict],
    reference: list[dict],
    path: list[tuple[int, int]],
    midi_path: str,
    output_path: Path,
) -> None:
    """
    Disegna predizione e spartito SOVRAPPOSTI sullo stesso asse temporale.

    Per ogni pitch le due sequenze occupano la stessa riga ma due metà
    diverse della barra (sopra = spartito, sotto = suonato): così lo
    scostamento orizzontale tra le due metà è lo SFASAMENTO temporale.
    Inoltre, per ogni coppia accoppiata dal DTW con stesso pitch, una linea
    sottile collega l'onset suonato a quello atteso, e il titolo riporta lo
    sfasamento medio/mediano.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    half = 0.42

    # Spartito: metà superiore della riga
    for n in reference:
        ax.broken_barh(
            [(n["time"], max(n.get("duration", 0.05), 0.02))],
            (_pitch(n) + 0.02, half),
            facecolors=REF_COLOR, alpha=0.55, edgecolors="white", linewidths=0.4,
        )
    # Suonato: metà inferiore della riga
    for n in predicted:
        ax.broken_barh(
            [(n["time"], max(n.get("duration", 0.05), 0.02))],
            (_pitch(n) - half - 0.02, half),
            facecolors=PRED_COLOR, alpha=0.6, edgecolors="white", linewidths=0.4,
        )

    # Linee di sfasamento (solo coppie con pitch uguale) + raccolta dei delta
    deltas = []
    for p_idx, r_idx in path:
        if p_idx >= len(predicted) or r_idx >= len(reference):
            continue
        p, r = predicted[p_idx], reference[r_idx]
        if _pitch(p) != _pitch(r):
            continue
        dt = p["time"] - r["time"]
        deltas.append(dt)
        ax.plot(
            [r["time"], p["time"]], [_pitch(r) + 0.02, _pitch(p) - 0.02],
            color=MISS_COLOR if abs(dt) > 0.1 else MATCH_COLOR,
            linewidth=1.0, alpha=0.7, zorder=3,
        )

    pitches = [_pitch(n) for n in predicted] + [_pitch(n) for n in reference] or [60]
    lo, hi = min(pitches) - 2, max(pitches) + 2
    ax.set_ylim(lo, hi)
    yticks = list(range(lo, hi + 1))
    ax.set_yticks(yticks)
    ax.set_yticklabels([midi_to_note_name(p) for p in yticks], fontsize=8)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Nota")
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    if deltas:
        import statistics
        mean_dt = statistics.mean(deltas)
        med_dt = statistics.median(deltas)
        shift_txt = f"sfasamento medio {mean_dt:+.3f}s (mediano {med_dt:+.3f}s)"
    else:
        shift_txt = "nessuna coppia con pitch uguale"

    ax.set_title(f"Sovrapposizione suonato/spartito — {Path(midi_path).name}\n{shift_txt}")

    legend_elems = [
        Patch(facecolor=REF_COLOR, alpha=0.55, label="spartito (MIDI)"),
        Patch(facecolor=PRED_COLOR, alpha=0.6, label="suonato (TabCNN)"),
        plt.Line2D([0], [0], color=MATCH_COLOR, label="sfasamento ≤ 0.1s"),
        plt.Line2D([0], [0], color=MISS_COLOR, label="sfasamento > 0.1s"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Overlay salvato:    {output_path}")


# =========================================================================
# Pipeline di confronto riutilizzabile (DTW + report + immagini)
# =========================================================================
def run_comparison(
    predicted: list[dict],
    reference: list[dict],
    midi_path: str,
    output_dir: str | None = None,
    stem: str = "comparison",
) -> list[tuple[int, int]]:
    """
    Allinea predizione e riferimento (DTW sul pitch), stampa il riepilogo e,
    se output_dir è fornito, salva due immagini:
        <stem>_vs_midi.png  → piano roll a due corsie con collegamenti DTW
        <stem>_overlay.png  → sovrapposizione sullo stesso asse (sfasamento)

    Returns:
        il path DTW (coppie pred_idx, ref_idx).
    """
    # DTW basato sul pitch: robusto a scale temporali diverse
    path = compute_dtw_alignment(predicted, reference, dist=pitch_only_distance)

    logger.info("")
    logger.info("=" * 56)
    logger.info("ACCOPPIAMENTO DTW (predetta → riferimento)")
    logger.info("=" * 56)
    logger.info(f"  {'pred':>16}   {'→':^3}   {'riferimento':>16}")
    for p_idx, r_idx in path:
        if p_idx >= len(predicted) or r_idx >= len(reference):
            continue
        p, r = predicted[p_idx], reference[r_idx]
        flag = "✓" if _pitch(p) == _pitch(r) else "✗"
        logger.info(
            f"  {p['time']:5.2f}s {midi_to_note_name(_pitch(p)):>4}   "
            f"{flag:^3}   "
            f"{r['time']:5.2f}s {midi_to_note_name(_pitch(r)):>4}"
        )

    # Metriche di accuratezza (note-level, frame-level, tempo/durata)
    metrics = compute_all_metrics(predicted, reference)
    print_metrics(metrics)

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("")
        plot_comparison(predicted, reference, path, midi_path, out_dir / f"{stem}_vs_midi.png")
        plot_overlay(predicted, reference, path, midi_path, out_dir / f"{stem}_overlay.png")
        save_metrics(metrics, out_dir / f"{stem}_metrics.json")

    return path


# =========================================================================
# MAIN
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visualizza le note predette (JSON di runTabCNN) accanto a uno spartito MIDI, allineate via DTW.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Esempio:\n"
            "  python -m guitar_tutor_pipeline.src.compareWithMidi \\\n"
            "    output_notes/cmaj/00_CMaj_120_cqt_notes.json SetTestHomeMade/00_CMaj_120.mid \\\n"
            "    --output_dir output_notes/cmaj\n"
        ),
    )
    parser.add_argument("predicted_json", type=str, help="JSON delle note predette (output di runTabCNN)")
    parser.add_argument("midi_path", type=str, help="File .mid/.midi dello spartito di riferimento")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Cartella dove salvare il piano roll .png. Se omesso, non salva nulla.",
    )
    args = parser.parse_args()

    if not Path(args.midi_path).exists():
        logger.error(f"File MIDI non trovato: {args.midi_path}")
        sys.exit(1)

    logger.info("🎯 Confronto predizione vs spartito (DTW)")
    logger.info(f"  Predizione: {args.predicted_json}")
    logger.info(f"  Spartito:   {args.midi_path}")
    logger.info("")

    predicted = load_predicted_notes(args.predicted_json)
    reference = parse_midi(args.midi_path)
    logger.info(f"  Note spartito caricate: {len(reference)}")

    run_comparison(
        predicted, reference, args.midi_path,
        output_dir=args.output_dir,
        stem=Path(args.predicted_json).stem,
    )

    logger.info("")
    logger.info("✅ Confronto completato")


if __name__ == "__main__":
    main()
