"""
model.py — Architettura CRNN per Automatic Music Transcription.

Implementazione fedele del modello 'Regress_onset_offset_frame_velocity_CRNN'
dal paper di Kong et al. (ByteDance):
  "High-resolution Piano Transcription with Pedals by Regressing Onset and Offset Times"

Questa architettura è compatibile con i pesi pre-addestrati sul dataset MAESTRO.
I nomi dei layer DEVONO corrispondere esattamente al checkpoint per il caricamento.

Componenti principali:
  1. Front-end: Spectrogram → LogMelFilterBank (via torchlibrosa)
  2. 4× AcousticModelCRnn indipendenti (frame, onset, offset, velocity)
  3. 2× GRU+FC di regressione per onset/offset ad alta risoluzione

Ogni AcousticModelCRnn contiene:
  - 4× ConvBlock (1→48→64→96→128 canali)
  - FC (3584→768) + BatchNorm1d
  - Bi-GRU (768→256×2=512)
  - FC (512→classes_num)
"""

import math
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

from .config import config

logger = logging.getLogger(__name__)

# =============================================================================
# Funzioni di inizializzazione pesi (identiche a Kong et al.)
# =============================================================================

def init_layer(layer):
    """Inizializza un layer Linear o Convoluzionale con Xavier uniform."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)

def init_bn(bn):
    """Inizializza un layer BatchNorm."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)

def init_gru(rnn):
    """Inizializza un layer GRU con schema specifico per AMT."""
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])

    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))

    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, "weight_ih_l{}".format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform],
        )
        torch.nn.init.constant_(getattr(rnn, "bias_ih_l{}".format(i)), 0)

        _concat_init(
            getattr(rnn, "weight_hh_l{}".format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_],
        )
        torch.nn.init.constant_(getattr(rnn, "bias_hh_l{}".format(i)), 0)

        if rnn.bidirectional:
            _concat_init(
                getattr(rnn, "weight_ih_l{}_reverse".format(i)),
                [_inner_uniform, _inner_uniform, _inner_uniform],
            )
            torch.nn.init.constant_(
                getattr(rnn, "bias_ih_l{}_reverse".format(i)), 0
            )
            _concat_init(
                getattr(rnn, "weight_hh_l{}_reverse".format(i)),
                [_inner_uniform, _inner_uniform, nn.init.orthogonal_],
            )
            torch.nn.init.constant_(
                getattr(rnn, "bias_hh_l{}_reverse".format(i)), 0
            )

# =============================================================================
# ConvBlock — Blocco convoluzionale base
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum); self.bn2 = nn.BatchNorm2d(out_channels, momentum)
        self.init_weight()
    def init_weight(self): init_layer(self.conv1); init_layer(self.conv2); init_bn(self.bn1); init_bn(self.bn2)
    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = F.relu_(self.bn1(self.conv1(input))); x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max": x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg": x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max": x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        return x

class AcousticModelCRnn(nn.Module):
    def __init__(self, classes_num, midfeat, momentum):
        super(AcousticModelCRnn, self).__init__()
        self.conv_block1 = ConvBlock(1, 48, momentum)
        self.conv_block2 = ConvBlock(48, 64, momentum)
        self.conv_block3 = ConvBlock(64, 96, momentum)
        self.conv_block4 = ConvBlock(96, 128, momentum)
        self.fc5 = nn.Linear(midfeat, 768, bias=False); self.bn5 = nn.BatchNorm1d(768, momentum)
        self.gru = nn.GRU(768, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, classes_num, bias=True)
        self.init_weight()
    def init_weight(self): init_layer(self.fc5); init_bn(self.bn5); init_gru(self.gru); init_layer(self.fc)
    def forward(self, input):
        x = self.conv_block1(input, pool_size=(1, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.transpose(1, 2).flatten(2)
        x = self.fc5(x); x = x.transpose(1, 2); x = self.bn5(x); x = x.transpose(1, 2)
        x = F.relu(x); x = F.dropout(x, p=0.5, training=self.training)
        x, _ = self.gru(x); x = self.fc(x)
        return x

class Regress_onset_offset_frame_velocity_CRNN(nn.Module):
    def __init__(self, frames_per_second=None, classes_num=None):
        super(Regress_onset_offset_frame_velocity_CRNN, self).__init__()
        if frames_per_second is None: frames_per_second = config.FRAMES_PER_SECOND
        if classes_num is None: classes_num = config.CLASSES_NUM
        sample_rate, window_size, hop_size = config.SAMPLE_RATE, config.WINDOW_SIZE, config.HOP_SIZE
        mel_bins, fmin, fmax = config.MEL_BINS, config.FMIN, config.FMAX
        momentum, midfeat = config.MOMENTUM, config.MIDFEAT

        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, win_length=window_size, window=config.WINDOW_TYPE, center=config.CENTER, pad_mode=config.PAD_MODE, freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)

        self.frame_model = AcousticModelCRnn(classes_num, midfeat, momentum)
        self.reg_onset_model = AcousticModelCRnn(classes_num, midfeat, momentum)
        self.reg_offset_model = AcousticModelCRnn(classes_num, midfeat, momentum)
        self.velocity_model = AcousticModelCRnn(classes_num, midfeat, momentum)

        self.reg_onset_gru = nn.GRU(input_size=classes_num * 2, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.reg_onset_fc = nn.Linear(512, classes_num, bias=True)
        self.frame_gru = nn.GRU(input_size=classes_num * 3, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.frame_fc = nn.Linear(512, classes_num, bias=True)
        self.init_weight()

    def init_weight(self): init_bn(self.bn0); init_gru(self.reg_onset_gru); init_layer(self.reg_onset_fc); init_gru(self.frame_gru); init_layer(self.frame_fc)
    
    def forward(self, input):
        x = self.spectrogram_extractor(input); x = self.logmel_extractor(x)
        x = x.transpose(1, 3); x = self.bn0(x); x = x.transpose(1, 3)

        frame_output = torch.sigmoid(self.frame_model(x))
        reg_onset_output = torch.sigmoid(self.reg_onset_model(x))
        reg_offset_output = torch.sigmoid(self.reg_offset_model(x))
        velocity_output = torch.sigmoid(self.velocity_model(x))

        x = torch.cat((reg_onset_output, (reg_onset_output ** 0.5) * velocity_output.detach()), dim=2)
        (x, _) = self.reg_onset_gru(x)
        x = F.dropout(x, p=0.5, training=self.training)
        reg_onset_output = torch.sigmoid(self.reg_onset_fc(x))

        x = torch.cat((frame_output, reg_onset_output.detach(), reg_offset_output.detach()), dim=2)
        (x, _) = self.frame_gru(x)
        x = F.dropout(x, p=0.5, training=self.training)
        frame_output = torch.sigmoid(self.frame_fc(x))

        output_dict = {
            "reg_onset_output": reg_onset_output,
            "reg_offset_output": reg_offset_output,
            "frame_output": frame_output,
            "velocity_output": velocity_output,
            "onset_output": reg_onset_output,
            "offset_output": reg_offset_output,
        }
        return output_dict

def load_maestro_checkpoint(model, checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    if "note_model" in state_dict: state_dict = state_dict["note_model"]
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    return model

def load_finetuned_checkpoint(model, checkpoint_path, device="cpu"):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")
    logger.info(f"Caricamento checkpoint fine-tuned da: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "?")
        val_loss = checkpoint.get("val_loss", "?")
        logger.info(f"Checkpoint epoca {epoch}, val_loss={val_loss}")
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def verify_checkpoint_compatibility(model, checkpoint_path, device="cpu"):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    missing = model_keys - checkpoint_keys
    unexpected = checkpoint_keys - model_keys

    result = {
        "model_keys": model_keys,
        "checkpoint_keys": checkpoint_keys,
        "missing": missing,
        "unexpected": unexpected,
        "compatible": len(missing) == 0,
    }
    if result["compatible"]:
        logger.info("✓ Checkpoint compatibile al 100% con il modello.")
    else:
        logger.warning(
            f"✗ Checkpoint parzialmente compatibile.\n"
            f"  Chiavi mancanti: {len(missing)}\n"
            f"  Chiavi inattese: {len(unexpected)}"
        )
    return result
