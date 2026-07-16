import argparse
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def generate_spectrograms(input_wav):
    """
    Generates and saves a CQT and a Log-Mel spectrogram for a given .wav file.
    Outputs are saved in the 'out' directory relative to the project root.
    """
    # Ensure input file exists
    if not os.path.isfile(input_wav):
        print(f"Error: The file {input_wav} does not exist.")
        return

    # Resolve project root and create 'out' directory
    # Assuming script is in 'scripts' folder, so parent is project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, 'scripts/out')
    os.makedirs(out_dir, exist_ok=True)

    # Base name for output files
    base_name = os.path.splitext(os.path.basename(input_wav))[0]
    
    print(f"Loading {input_wav}...")
    # Load audio (sr=None preserves original sample rate)
    y, sr = librosa.load(input_wav, sr=None)

    # --- 1. Compute and plot CQT Spectrogram (TabCNN config) ---
    print("Computing CQT Spectrogram (TabCNN style)...")
    sr_cqt = 22050
    y_cqt = librosa.resample(y, orig_sr=sr, target_sr=sr_cqt)
    
    hop_length_cqt = 512
    n_bins_cqt = 192
    bins_per_octave_cqt = 24
    fmin_cqt = 32.70319566257483  # C1
    
    cqt = librosa.cqt(y_cqt, sr=sr_cqt, hop_length=hop_length_cqt, fmin=fmin_cqt, n_bins=n_bins_cqt, bins_per_octave=bins_per_octave_cqt)
    cqt_mag = np.abs(cqt)
    cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)
    cqt_norm = (cqt_db / 80.0) + 1.0  # Pipeline normalization

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(cqt_norm, sr=sr_cqt, hop_length=hop_length_cqt, x_axis='time', y_axis='cqt_hz', 
                             fmin=fmin_cqt, bins_per_octave=bins_per_octave_cqt, cmap='magma')
    plt.colorbar(format='%+2.2f Norm')
    plt.title(f'CQT Spectrogram (TabCNN: 192 bins, 24/oct) - {base_name}')
    plt.tight_layout()
    
    cqt_out_path = os.path.join(out_dir, f"{base_name}_cqt.png")
    plt.savefig(cqt_out_path, dpi=150)
    plt.close()
    print(f"✅ Saved CQT image to: {cqt_out_path}")

    # --- 2. Compute and plot Log-Mel Spectrogram (CRNN MAESTRO config) ---
    print("Computing Log-Mel Spectrogram (CRNN style)...")
    sr_mel = 16000
    y_mel = librosa.resample(y, orig_sr=sr, target_sr=sr_mel)
    
    hop_length_mel = 160
    n_fft_mel = 2048
    n_mels = 229
    fmin_mel = 30
    fmax_mel = sr_mel // 2
    
    # Compute spectrogram magnitude first
    stft = librosa.stft(y_mel, n_fft=n_fft_mel, hop_length=hop_length_mel, window='hann', center=True)
    mag_spec = np.abs(stft)
    
    # Compute Mel filterbank
    mel_basis = librosa.filters.mel(sr=sr_mel, n_fft=n_fft_mel, n_mels=n_mels, fmin=fmin_mel, fmax=fmax_mel)
    mel_spec = np.dot(mel_basis, mag_spec)
    
    # Log scaling (like torchlibrosa: log10 or simply log with amin)
    log_mel_spec = np.log10(mel_spec + 1e-10)

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(log_mel_spec, sr=sr_mel, hop_length=hop_length_mel, x_axis='time', y_axis='mel', 
                             fmin=fmin_mel, fmax=fmax_mel, cmap='magma')
    plt.colorbar(format='%+2.2f Log')
    plt.title(f'Log-Mel Spectrogram (CRNN: 229 mels) - {base_name}')
    plt.tight_layout()
    
    mel_out_path = os.path.join(out_dir, f"{base_name}_logmel.png")
    plt.savefig(mel_out_path, dpi=150)
    plt.close()
    print(f"✅ Saved Log-Mel image to: {mel_out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera due immagini (CQT e Log-Mel) partendo da un file .wav")
    parser.add_argument("input_wav", help="Percorso del file .wav di input")
    
    args = parser.parse_args()
    generate_spectrograms(args.input_wav)
