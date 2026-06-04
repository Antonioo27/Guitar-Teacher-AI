"""
model.py — Modulo 2: Definizione dell'architettura TabCNN e caricamento pesi.

TabCNN (Tablature CNN) è una rete neurale convoluzionale progettata per la
trascrizione automatica di chitarra. Per ogni frame temporale della CQT,
il modello predice quale tasto è premuto su ciascuna delle 6 corde.

Architettura ispirata a:
  Wiggins, A. (2019). "Guitar Tablature Estimation with a Convolutional
  Neural Network". ISMIR.

L'architettura usa tre blocchi convoluzionali con BatchNorm e Dropout,
seguiti da teste FC separate per ogni corda (multi-output classification).

NOTA: La definizione dell'architettura DEVE corrispondere esattamente a
quella usata nel training su Colab. Se i pesi non si caricano, verificare
che le dimensioni dei layer corrispondano.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config

logger = logging.getLogger(__name__)


class TabCNN(nn.Module):
    """
    Rete Convoluzionale per Guitar Tablature Transcription.

    Per ogni frame temporale (con contesto), predice per ciascuna delle
    6 corde della chitarra quale tasto è premuto (0-19) o se la corda
    non è suonata (classe speciale).

    Input:
        Tensore di shape (batch, 1, n_bins, context_frames)
        - n_bins: numero di bin CQT (default: 192)
        - context_frames: finestra temporale locale (default: 9)

    Output:
        Lista di 6 tensori, ciascuno di shape (batch, num_classes)
        dove num_classes = NUM_FRETS + 1 = 21
    """

    def __init__(
        self,
        n_bins: int = config.N_BINS,
        context_frames: int = config.CONTEXT_FRAMES,
        num_strings: int = config.NUM_STRINGS,
        num_classes: int = config.NUM_CLASSES,
        dropout_rate: float = 0.25,
    ):
        super().__init__()

        self.num_strings = num_strings
        self.num_classes = num_classes

        # =====================================================================
        # Blocchi Convoluzionali
        # =====================================================================

        # Blocco 1: estrae feature timbriche locali
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)

        # Blocco 2: feature a medio raggio
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)

        # Blocco 3: feature ad alto livello
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout_conv = nn.Dropout2d(p=dropout_rate)

        # =====================================================================
        # Calcolo delle dimensioni dopo le convoluzioni
        # =====================================================================
        # Dopo 3 layer di MaxPool2d(2,1) sulla dimensione freq:
        # n_bins: 192 → 96 → 48 → 24
        # context_frames resta invariato (pool solo su freq)
        self._flat_size = 64 * (n_bins // 8) * context_frames

        # =====================================================================
        # Layer Fully Connected condiviso
        # =====================================================================
        self.fc_shared = nn.Linear(self._flat_size, 256)
        self.dropout_fc = nn.Dropout(p=0.5)

        # =====================================================================
        # Teste di output: una per ogni corda
        # =====================================================================
        # Ogni testa produce num_classes logits (classificazione multi-classe)
        self.string_heads = nn.ModuleList([
            nn.Linear(256, num_classes) for _ in range(num_strings)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor di shape (batch, 1, n_bins, context_frames).

        Returns:
            Lista di 6 tensori (uno per corda), ciascuno di shape
            (batch, num_classes) con i logits per la classificazione
            del tasto premuto.
        """
        # Blocco Conv 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(x)

        # Blocco Conv 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)

        # Blocco Conv 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout_conv(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC condiviso
        x = F.relu(self.fc_shared(x))
        x = self.dropout_fc(x)

        # Output per-corda
        outputs = [head(x) for head in self.string_heads]

        return outputs


def load_model(
    weights_path: str | Path = config.WEIGHTS_PATH,
    device: str = "cpu",
) -> TabCNN:
    """
    Carica il modello TabCNN con i pesi pre-addestrati.

    Args:
        weights_path: Percorso al file dei pesi (.pt o .pth).
        device: Device su cui caricare il modello ("cpu" o "cuda").

    Returns:
        Modello TabCNN in evaluation mode.

    Raises:
        FileNotFoundError: Se il file dei pesi non esiste.
        RuntimeError: Se i pesi non corrispondono all'architettura.
    """
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"File dei pesi non trovato: {weights_path}\n"
            f"Assicurati che il file '{config.WEIGHTS_FILENAME}' sia presente "
            f"nella directory: {config.WEIGHTS_DIR}"
        )

    logger.info(f"Caricamento modello TabCNN da: {weights_path}")

    model = TabCNN()

    # Carica i pesi (compatibile sia con state_dict che con checkpoint completi)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Checkpoint salvato con informazioni aggiuntive (optimizer, epoch, ecc.)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Checkpoint caricato — Epoch: {checkpoint.get('epoch', '?')}, "
            f"Loss: {checkpoint.get('loss', '?')}"
        )
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # File contiene direttamente lo state_dict
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    logger.info(f"Modello caricato con successo su device: {device}")
    return model
