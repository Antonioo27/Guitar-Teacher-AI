import os
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast

from .config import config

class TranscriptionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output_dict, target_dict):
        with autocast(device_type='cuda', enabled=False):
            frame_output      = output_dict["frame_output"].float()
            reg_onset_output  = output_dict["reg_onset_output"].float()
            reg_offset_output = output_dict["reg_offset_output"].float()
            velocity_output   = output_dict["velocity_output"].float()

            frame_target      = target_dict["frame_target"].float()
            reg_onset_target  = target_dict["reg_onset_target"].float()
            reg_offset_target = target_dict["reg_offset_target"].float()
            velocity_target   = target_dict["velocity_target"].float()
            onset_target      = target_dict["onset_target"].float()

            def bce(output, target):
                output = torch.clamp(output, 1e-7, 1.0 - 1e-7)
                return -target * torch.log(output) - (1.0 - target) * torch.log(1.0 - output)

            frame_loss      = torch.mean(bce(frame_output, frame_target))
            reg_onset_loss  = torch.mean(bce(reg_onset_output, reg_onset_target))
            reg_offset_loss = torch.mean(bce(reg_offset_output, reg_offset_target))

            v_mask = onset_target
            v_loss_raw = bce(velocity_output, velocity_target)
            velocity_loss = (v_loss_raw * v_mask).sum() / v_mask.sum().clamp(min=1.0)

            return reg_onset_loss + reg_offset_loss + frame_loss + velocity_loss

def build_optimizer_and_scheduler(model, total_steps):
    """AdamW + step decay 0.9 spalmato su 10 tappe lungo l'intero training."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    step_size = max(1, total_steps // 10)
    def lr_lambda(current_step):
        return 0.9 ** (current_step / step_size)

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler

def save_checkpoint(epoch, model, optimizer, scheduler, scaler, best_val_loss, patience_counter):
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter
    }, config.CHECKPOINT_PATH)

def load_checkpoint(model, optimizer, scheduler, scaler, device):
    if os.path.exists(config.CHECKPOINT_PATH):
        print("Trovato checkpoint esistente. Riprendendo il training...")
        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        print(f"  -> Ripreso dall'epoch {start_epoch}. Best Val Loss: {best_val_loss:.4f}")
        return start_epoch, best_val_loss, patience_counter
    else:
        print("Nessun checkpoint trovato. Partenza da zero.")
        return 1, float('inf'), 0

def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(loader):
        waveform = batch["waveform"].to(device, non_blocking=True)
        target_dict = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "waveform"}

        optimizer.zero_grad(set_to_none=True)

        output_dict = model(waveform)
        loss = loss_fn(output_dict, target_dict)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
    return total_loss / len(loader)

def run_training():
    import random
    from huggingface_hub import snapshot_download
    from torch.utils.data import DataLoader
    from .dataset import GapsDataset
    from .model import Regress_onset_offset_frame_velocity_CRNN, load_maestro_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device attivo: {device}")

    # Scarica GAPS
    if not config.DATA_DIR.exists():
        print("Scaricamento GAPS dataset...")
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id="xavriley/GAPS", repo_type="dataset",
                          local_dir=str(config.DATA_DIR), allow_patterns=["audio/*.wav", "midi/*.mid"])

    audio_dir = config.DATA_DIR / "audio"
    midi_dir = config.DATA_DIR / "midi"
    gaps_pairs = [(audio_dir / f.name.replace(".mid", ".wav"), f) for f in midi_dir.glob("*.mid")]
    gaps_pairs = [(a, m) for a, m in gaps_pairs if a.exists()]
    print(f"Trovate {len(gaps_pairs)} coppie audio/midi GAPS.")

    random.shuffle(gaps_pairs)
    n_total = len(gaps_pairs)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    train_pairs = gaps_pairs[:n_train]
    val_pairs = gaps_pairs[n_train:n_train+n_val]
    test_pairs = gaps_pairs[n_train+n_val:]

    train_ds = GapsDataset(train_pairs, config.SEGMENT_SECONDS, hop_seconds=1.0, augment=True)
    val_ds = GapsDataset(val_pairs, config.SEGMENT_SECONDS, hop_seconds=config.SEGMENT_SECONDS, augment=False)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("Inizializzazione modello...")
    model = Regress_onset_offset_frame_velocity_CRNN(
        frames_per_second=config.FRAMES_PER_SECOND,
        classes_num=config.CLASSES_NUM
    ).to(device)

    # Assicurati che il checkpoint esista (scaricalo se necessario)
    import urllib.request
    if not config.MAESTRO_CHECKPOINT.exists() or os.path.getsize(config.MAESTRO_CHECKPOINT) < 1_000_000:
        print("Download dei pesi MAESTRO da Zenodo in corso...")
        urllib.request.urlretrieve("https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1", str(config.MAESTRO_CHECKPOINT))
        print(f"✓ Download completato! Dimensione: {os.path.getsize(config.MAESTRO_CHECKPOINT) / 1e6:.2f} MB")
        
    model = load_maestro_checkpoint(model, config.MAESTRO_CHECKPOINT, device=device)
    print("✅ Checkpoint MAESTRO caricato.")

    for param in model.parameters():
        param.requires_grad = True

    total_steps = len(train_loader) * config.NUM_EPOCHS
    optimizer, scheduler = build_optimizer_and_scheduler(model, total_steps)
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None
    loss_fn = TranscriptionLoss()

    start_epoch, best_val_loss, patience_counter = load_checkpoint(model, optimizer, scheduler, scaler, device)

    print(f"\nInizio Training da Epoch {start_epoch} fino a {config.NUM_EPOCHS}...")
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.NUM_EPOCHS} ---")
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, device, scaler)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                waveform = batch["waveform"].to(device, non_blocking=True)
                target_dict = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "waveform"}
                output_dict = model(waveform)
                val_loss += loss_fn(output_dict, target_dict).item()
        val_loss /= len(val_loader)

        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch} riassunto | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print("  -> ✅ Nuovo best model salvato!")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  -> Val Loss non migliorata. Patience: {patience_counter}/{config.PATIENCE}")
            if patience_counter >= config.PATIENCE:
                print("  -> ⚠️ Early stopping attivato!")
                save_checkpoint(epoch, model, optimizer, scheduler, scaler, best_val_loss, patience_counter)
                break

        save_checkpoint(epoch, model, optimizer, scheduler, scaler, best_val_loss, patience_counter)

    print("\n🎉 Training completato!")

if __name__ == "__main__":
    run_training()
