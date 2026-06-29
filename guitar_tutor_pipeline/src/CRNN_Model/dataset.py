import os
import random
import numpy as np
import torch
import librosa
import pretty_midi
from torch.utils.data import Dataset, DataLoader
from .config import config

def parse_midi_to_events(midi_path):
    """Legge un file MIDI e restituisce una lista di eventi nota ordinati."""
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    events = []
    for instrument in midi_data.instruments:
        if instrument.is_drum: continue
        for note in instrument.notes:
            events.append({
                "start": float(note.start),
                "end": float(note.end),
                "pitch": int(note.pitch),
                "velocity": int(note.velocity)
            })
    events.sort(key=lambda x: x["start"])
    return events

def get_regression(center_frame_float, num_frames, left_frames=3, right_frames=3):
    """Rampa che codifica lo scarto sub-frame (stile Kong)."""
    target = np.zeros(num_frames, dtype=np.float32)
    step = 1.0 / (right_frames + 1)

    center_int = int(round(center_frame_float))
    shift = center_frame_float - center_int     # ora in [-0.5, +0.5], coerente col centro

    for i in range(-left_frames, right_frames + 1):
        idx = center_int + i
        if 0 <= idx < num_frames:
            dist = abs(i - shift)               # distanza dal vero onset, non dal frame arrotondato
            target[idx] = max(0.0, 1.0 - dist * step)
    return target

def create_targets_from_midi_events(midi_events, num_frames, classes_num, begin_note, frames_per_second):
    frame_target = np.zeros((num_frames, classes_num), dtype=np.float32)
    onset_target = np.zeros((num_frames, classes_num), dtype=np.float32)
    offset_target = np.zeros((num_frames, classes_num), dtype=np.float32)
    velocity_target = np.zeros((num_frames, classes_num), dtype=np.float32)
    reg_onset_target = np.zeros((num_frames, classes_num), dtype=np.float32)
    reg_offset_target = np.zeros((num_frames, classes_num), dtype=np.float32)

    for event in midi_events:
        pitch = event["pitch"] - begin_note
        if pitch < 0 or pitch >= classes_num: continue

        # Usa i float per calcolare la posizione esatta sub-frame
        start_frame_float = event["start"] * frames_per_second
        end_frame_float = event["end"] * frames_per_second
        start_frame = int(round(start_frame_float))
        end_frame = int(round(end_frame_float))
        start_frame = max(0, min(num_frames - 1, start_frame))
        end_frame = max(0, min(num_frames - 1, end_frame))
        if start_frame == end_frame: end_frame = min(num_frames - 1, start_frame + 1)

        frame_target[start_frame : end_frame + 1, pitch] = 1.0
        onset_target[start_frame, pitch] = 1.0
        offset_target[end_frame, pitch] = 1.0

        # Velocity: /128 e messa solo sul frame di onset (come Kong)
        velocity_target[start_frame, pitch] = event["velocity"] / 128.0

        # Regression sub-frame (asimmetriche, dipendenti dallo shift)
        reg_onset_target[:, pitch] = get_regression(start_frame_float, num_frames)
        reg_offset_target[:, pitch] = get_regression(end_frame_float, num_frames)

    return {
        "frame_target": frame_target, "onset_target": onset_target, "offset_target": offset_target,
        "velocity_target": velocity_target, "reg_onset_target": reg_onset_target, "reg_offset_target": reg_offset_target
    }

class GapsDataset(Dataset):
    def __init__(self, pairs, segment_seconds, hop_seconds, augment=False):
        self.pairs = pairs
        self.segment_seconds = segment_seconds
        self.hop_seconds = hop_seconds
        self.augment = augment
        self.segment_samples = int(segment_seconds * config.SAMPLE_RATE)
        # FIX: num_output_frames a 100 fps (1000 frame per 10 secondi)
        self.num_output_frames = round(segment_seconds * config.OUTPUT_FPS) + 1
        self.segments = self._build_segment_index()

    def _build_segment_index(self):
        segments = []
        for pair_idx, (audio_path, _) in enumerate(self.pairs):
            try: duration = librosa.get_duration(path=str(audio_path))
            except: continue
            if duration < self.segment_seconds: segments.append((pair_idx, 0.0))
            else:
                start = 0.0
                while start + self.segment_seconds <= duration:
                    segments.append((pair_idx, start))
                    start += self.hop_seconds
        return segments

    def __len__(self): return len(self.segments)

    def __getitem__(self, idx):
        pair_idx, start_time = self.segments[idx]
        audio_path, midi_path = self.pairs[pair_idx]
        waveform, _ = librosa.load(str(audio_path), sr=config.SAMPLE_RATE, mono=True, offset=start_time, duration=self.segment_seconds)
        if len(waveform) < self.segment_samples: waveform = np.pad(waveform, (0, self.segment_samples - len(waveform)))
        elif len(waveform) > self.segment_samples: waveform = waveform[:self.segment_samples]

        events = parse_midi_to_events(midi_path)
        segment_events = [{"start": max(0, e["start"] - start_time), "end": min(self.segment_seconds, e["end"] - start_time),
                           "pitch": e["pitch"], "velocity": e["velocity"]} for e in events if e["end"] > start_time and e["start"] < start_time + self.segment_seconds]

        targets = create_targets_from_midi_events(segment_events, self.num_output_frames, config.CLASSES_NUM, config.BEGIN_NOTE, config.OUTPUT_FPS)

        if self.augment:
            shift = random.randint(-2, 2)
            if shift != 0:
                waveform = librosa.effects.pitch_shift(waveform, sr=config.SAMPLE_RATE, n_steps=shift)
                for k in targets.keys(): targets[k] = np.roll(targets[k], shift, axis=1)
                if shift > 0:
                    for k in targets.keys(): targets[k][:, :shift] = 0
                elif shift < 0:
                    for k in targets.keys(): targets[k][:, shift:] = 0
            if random.random() < 0.5:
                gain_db = random.uniform(-6.0, 6.0); waveform = waveform * (10.0 ** (gain_db / 20.0))
            waveform = np.clip(waveform, -1.0, 1.0)

        sample = {"waveform": torch.from_numpy(waveform).float()}
        for k in targets: sample[k] = torch.from_numpy(targets[k]).float()
        return sample
