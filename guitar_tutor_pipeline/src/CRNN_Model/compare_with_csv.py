# compare_with_csv.py
# Confronta un MIDI predetto con una ground truth in CSV (time,duration,midi_pitch,...)
# Uso:
#   python -m guitar_tutor_pipeline.src.CRNN_Model.compare_with_csv <predetto.mid> <ground_truth.csv>
import sys, csv, argparse
from pathlib import Path
import pretty_midi


def load_csv(path):
    ref = []
    with open(path) as f:
        for r in csv.DictReader(f):
            s = float(r["time"]); d = float(r["duration"]); p = int(r["midi_pitch"])
            ref.append((s, s + d, p))
    return sorted(ref)


def load_midi(path):
    pm = pretty_midi.PrettyMIDI(str(path))
    return sorted((n.start, n.end, n.pitch)
                  for inst in pm.instruments for n in inst.notes)


def name(p):
    return pretty_midi.note_number_to_name(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mid", type=Path, help="MIDI predetto")
    ap.add_argument("csv", type=Path, help="ground truth CSV")
    ap.add_argument("--tol", type=float, default=0.05, help="tolleranza onset (s)")
    args = ap.parse_args()

    ref = load_csv(args.csv)
    est = load_midi(args.mid)
    TOL = args.tol

    print(f"GROUND TRUTH: {len(ref)} note   |   PREDIZIONE: {len(est)} note\n")

    used = [False] * len(est)
    matched, missed = [], []
    for i, (rs, re, rp) in enumerate(ref):
        best, bestdt = None, 1e9
        for j, (es, ee, ep) in enumerate(est):
            if used[j] or ep != rp:
                continue
            dt = abs(es - rs)
            if dt <= TOL and dt < bestdt:
                best, bestdt = j, dt
        if best is not None:
            used[best] = True; matched.append((i, best))
        else:
            missed.append(i)
    extra = [j for j in range(len(est)) if not used[j]]

    print("== NOTE CORRETTE (onset±%dms + pitch esatto) ==" % (TOL * 1000))
    print(f"{'ref onset':>9} {'pitch':>5}   {'est onset':>9}  {'d_onset(ms)':>11}")
    for i, j in sorted(matched, key=lambda m: ref[m[0]][0]):
        rs, re, rp = ref[i]; es, ee, ep = est[j]
        print(f"{rs:9.3f} {name(rp):>5}   {es:9.3f}  {(es-rs)*1000:+11.1f}")

    print(f"\n== NOTE MANCANTI (in GT, non predette)  [{len(missed)}] ==")
    for i in missed:
        rs, re, rp = ref[i]
        print(f"  t={rs:6.3f}s  {name(rp):>4} (pitch {rp})  dur={re-rs:.2f}s")

    print(f"\n== FALSI POSITIVI (predette, non in GT)  [{len(extra)}] ==")
    for j in extra:
        es, ee, ep = est[j]
        print(f"  t={es:6.3f}s  {name(ep):>4} (pitch {ep})  dur={ee-es:.2f}s")

    TP, FP, FN = len(matched), len(extra), len(missed)
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    print("\n" + "=" * 46)
    print(f"TP={TP}  FP={FP}  FN={FN}")
    print(f"Precision={P*100:.1f}%  Recall={R*100:.1f}%  F1={F*100:.1f}%")
    print("=" * 46)


if __name__ == "__main__":
    main()
