"""
model.py — Modulo 2: Definizione dell'architettura TabCNN e caricamento pesi.

Implementazione adattata dal notebook del collega, che utilizza amt_tools per
gestire la struttura del modello e i raggruppamenti Softmax.
"""

import sys
import logging
from pathlib import Path

import torch
from torch import nn

from amt_tools.models.common import TranscriptionModel, SoftmaxGroups
from amt_tools import tools
from amt_tools.tools.instrument import GuitarProfile

from . import config

logger = logging.getLogger(__name__)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ResBlock, self).__init__()
        padding = kernel_size // 2  # Preserve spatial dimensions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batchnorm(out)
        # put shortcut through fully connected layer to match dimensions
        shortcut = self.shortcut(shortcut)
        out += shortcut

        return out


class TabCNN(TranscriptionModel):
    """
    Implements the TabCNN model (http://archives.ismir.net/ismir2019/paper/000033.pdf).
    Eredita da TranscriptionModel di amt_tools.
    """

    def __init__(self, dim_in, profile, in_channels=1, model_complexity=1, device='cpu'):
        """
        Initialize the model and establish parameter defaults in function signature.
        """
        super().__init__(dim_in, profile, in_channels, model_complexity, 9, device)

        # Initialize a flag to check whether to pad input features
        self.online = False

        # Number of filters for each stage
        nf1 = 32 * self.model_complexity
        nf2 = 64 * self.model_complexity
        nf3 = nf2

        # Kernel size for each stage
        ks1 = (3, 3)
        ks2 = ks1
        ks3 = ks1

        # Reduction size for each stage
        rd1 = (2, 2)

        # Dropout percentages for each stage
        dp1 = 0.25
        dp2 = 0.50

        self.conv = nn.Sequential(
            # 1st convolution
            nn.Conv2d(self.in_channels, nf1, ks1),
            # Activation function
            nn.ReLU(inplace=True),
            # 2nd convolution
            nn.Conv2d(nf1, nf2, ks2),
            # Activation function
            nn.ReLU(inplace=True),
            # 3rd convolution
            nn.Conv2d(nf2, nf3, ks3),
            # Activation function
            nn.ReLU(inplace=True),
            # 1st reduction
            nn.MaxPool2d(rd1),
            # 1st dropout
            nn.Dropout(dp1)
        )

        # Determine the height, width, and total size of the feature map
        feat_map_height = (self.dim_in - 6) // 2
        feat_map_width = (self.frame_width - 6) // 2
        self.conv_embedding_size = nf3 * feat_map_height * feat_map_width

        # Number of neurons for each fully-connected stage
        self.fc_embedding_size = 128 * self.model_complexity

        # Extract tablature parameters
        num_groups = self.profile.get_num_dofs()
        num_classes = self.profile.num_pitches + 1

        self.dense = nn.Sequential(
            # 1st fully-connected
            nn.Linear(self.conv_embedding_size, self.fc_embedding_size),
            # Activation function
            nn.ReLU(inplace=True),
            # 2nd dropout
            nn.Dropout(dp2),
            # 2nd fully-connected
            SoftmaxGroups(self.fc_embedding_size, num_groups, num_classes)
        )

    def toggle_online(self):
        self.online = not self.online

    def pre_proc(self, batch):
        batch = super().pre_proc(batch)

        # Extract the features from the batch as a NumPy array
        feats = tools.tensor_to_array(batch[tools.KEY_FEATS])
        # Window the features to mimic online/real-time operation
        feats = tools.framify_activations(feats, self.frame_width, pad=(not self.online))
        # Convert the features back to PyTorch tensor and add to device
        feats = tools.array_to_tensor(feats, self.device)
        # Switch the sequence-frame and feature axes
        feats = feats.transpose(-2, -3)
        # Switch the sequence-frame and channel axes
        feats = feats.transpose(-3, -4)

        batch[tools.KEY_FEATS] = feats

        return batch

    def forward(self, feats):
        # Initialize an empty dictionary to hold output
        output = dict()

        # Obtain the batch size before sequence-frame axis is collapsed
        batch_size = feats.size(0)

        # Collapse the sequence-frame axis into the batch axis
        feats = feats.reshape(-1, self.in_channels, self.dim_in, self.frame_width)

        # Obtain the feature embeddings
        embeddings = self.conv(feats)
        # Flatten spatial features into one embedding
        embeddings = embeddings.flatten(1)
        # Size of the embedding
        embedding_size = embeddings.size(-1)
        # Restore proper batch dimension, unsqueezing sequence-frame axis
        embeddings = embeddings.view(batch_size, -1, embedding_size)

        # Obtain the tablature estimate and add it to the output dictionary
        output[tools.KEY_TABLATURE] = self.dense(embeddings)

        return output

    def post_proc(self, batch):
        # Extract the raw output
        output = batch[tools.KEY_OUTPUT]

        # Obtain pointers to the output layer
        tablature_output_layer = self.dense[-1]

        # Obtain the tablature estimation
        tablature_est = output[tools.KEY_TABLATURE]

        # Finalize tablature estimation
        output[tools.KEY_TABLATURE] = tablature_output_layer.finalize_output(tablature_est)

        return output


def load_model(
    weights_path: str | Path = config.WEIGHTS_PATH,
    device: str = "cpu",
) -> TabCNN:
    """
    Carica il modello TabCNN con i pesi pre-addestrati, risolvendo
    i problemi di referenze globali del checkpoint.
    """
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"File dei pesi non trovato: {weights_path}\n"
            f"Assicurati che il file '{config.WEIGHTS_FILENAME}' sia presente "
            f"nella directory: {config.WEIGHTS_DIR}"
        )

    logger.info(f"Caricamento modello TabCNN da: {weights_path}")

    # Inizializza il profilo strumento (6 corde standard)
    profile = GuitarProfile()

    # Inizializza l'architettura
    model = TabCNN(dim_in=config.N_BINS, profile=profile, in_channels=1, model_complexity=4, device=device)

    # TRUCCO: Mappa il modulo corrente a 'tabcnn' su sys.modules
    # in modo che il caricamento del checkpoint (salvato come tabcnn.TabCNN)
    # riesca a mappare la classe a questo modulo.
    sys.modules['tabcnn'] = sys.modules[__name__]

    try:
        # Pesi salvati con weights_only=False necessario per via della serializzazione personalizzata
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, torch.nn.Module):
            model.load_state_dict(checkpoint.state_dict())
            logger.info("Estratto lo state_dict dall'oggetto modello salvato.")
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Estratto model_state_dict dal dizionario.")
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
            logger.info("Pesi caricati direttamente dallo state_dict.")

    except Exception as e:
        logger.error(f"Errore durante il caricamento dei pesi: {e}")
        raise

    model.to(device)
    model.eval()

    logger.info(f"Modello caricato con successo su device: {device}")
    return model
