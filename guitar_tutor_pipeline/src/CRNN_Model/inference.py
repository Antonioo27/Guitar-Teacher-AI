import numpy as np
import torch
import librosa
import pretty_midi
import mir_eval

from .config import config
from .model import Regress_onset_offset_frame_velocity_CRNN

@torch.no_grad()
def transcribe_full(model, wav_path, device, onset_thresh=0.3, frame_thresh=0.3,
                    segment_seconds=10.0, hop_seconds=10.0):
    """Trascrive un file intero a segmenti, decodificando con interpolazione sub-frame."""
    model.eval()
    fps = config.OUTPUT_FPS  # 100
    audio, _ = librosa.load(str(wav_path), sr=config.SAMPLE_RATE, mono=True)
    seg_samples = int(segment_seconds * config.SAMPLE_RATE)

    notes = []
    start = 0.0
    while start < len(audio) / config.SAMPLE_RATE:
        s0 = int(start * config.SAMPLE_RATE)
        chunk = audio[s0 : s0 + seg_samples]
        if len(chunk) < seg_samples:
            chunk = np.pad(chunk, (0, seg_samples - len(chunk)))
        wav_t = torch.from_numpy(chunk).float().unsqueeze(0).to(device)

        out = model(wav_t)
        onset = out["reg_onset_output"][0].cpu().numpy()    # (T, 88) rampe
        frame = out["frame_output"][0].cpu().numpy()
        vel   = out["velocity_output"][0].cpu().numpy()

        T, P = onset.shape
        for p in range(P):
            t = 1
            while t < T - 1:
                # picco locale sulla rampa di regressione
                if onset[t, p] >= onset_thresh and onset[t, p] >= onset[t-1, p] and onset[t, p] > onset[t+1, p]:
                    # interpolazione parabolica per il tempo sub-frame
                    a, b, c = onset[t-1, p], onset[t, p], onset[t+1, p]
                    denom = (a - 2*b + c)
                    shift = 0.5 * (a - c) / denom if abs(denom) > 1e-6 else 0.0
                    onset_time = start + (t + shift) / fps

                    # durata: estendi finché frame resta attivo
                    end_t = t + 1
                    while end_t < T and frame[end_t, p] > frame_thresh:
                        end_t += 1
                    end_time = start + end_t / fps
                    if end_time <= onset_time:
                        end_time = onset_time + 0.05

                    velocity = int(np.clip(vel[t, p] * 128, 1, 127))
                    notes.append((onset_time, end_time, p + config.BEGIN_NOTE, velocity))
                    t += 2  # refrattarietà
                else:
                    t += 1
        start += hop_seconds

    return notes

def evaluate_model(model, test_pairs, device):
    all_P, all_R, all_F = [], [], []
    print(f"{'file':<22} {'P':>6} {'R':>6} {'F1':>6} {'ref':>5} {'est':>5}")
    for wav_path, midi_path in test_pairs:
        est = transcribe_full(model, wav_path, device)
        ref_pm = pretty_midi.PrettyMIDI(str(midi_path))
        ref = [(n.start, n.end, n.pitch) for inst in ref_pm.instruments
               for n in inst.notes if not inst.is_drum]
        if len(ref) == 0 or len(est) == 0:
            continue

        ref_int = np.array([[s, e] for s, e, _ in ref])
        ref_pit = np.array([p for _, _, p in ref])
        est_int = np.array([[s, e] for s, e, _, _ in est])
        est_pit = np.array([p for _, _, p, _ in est])

        P, R, F1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_int, ref_pit, est_int, est_pit,
            onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
        all_P.append(P); all_R.append(R); all_F.append(F1)
        print(f"{wav_path.name:<22} {P*100:6.1f} {R*100:6.1f} {F1*100:6.1f} {len(ref):5d} {len(est):5d}")

    print("\n" + "="*48)
    print(f"  MEDIA TEST SET ({len(all_F)} file)")
    print(f"  Precision: {np.mean(all_P)*100:.2f}%  Recall: {np.mean(all_R)*100:.2f}%  F1: {np.mean(all_F)*100:.2f}%")
    print("="*48)
