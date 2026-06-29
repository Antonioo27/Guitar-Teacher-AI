import os
from pathlib import Path

class Config:
    # Paths
    WORK_DIR = Path(__file__).parent.resolve()
    PIPELINE_ROOT = WORK_DIR.parent.parent
    DATA_DIR = PIPELINE_ROOT / "data" / "gaps_data"
    GS_DATA_DIR = PIPELINE_ROOT / "data" / "guitarset_data"
    WEIGHTS_DIR = PIPELINE_ROOT / "weights"
    MAESTRO_CHECKPOINT = WEIGHTS_DIR / "MaestroModel.pth"
    BEST_MODEL_PATH = WEIGHTS_DIR / "best_guitar_model.pth"
    CHECKPOINT_DIR = WEIGHTS_DIR / "checkpoints"
    CHECKPOINT_PATH = CHECKPOINT_DIR / "training_checkpoint.pth"

    # Audio
    SAMPLE_RATE = 16000
    WINDOW_SIZE = 2048
    HOP_SIZE = 160
    MEL_BINS = 229
    FMIN = 30
    FMAX = SAMPLE_RATE // 2
    WINDOW_TYPE = "hann"
    CENTER = True
    PAD_MODE = "reflect"

    # Model
    CLASSES_NUM = 88
    BEGIN_NOTE = 21
    FRAMES_PER_SECOND = SAMPLE_RATE // HOP_SIZE
    OUTPUT_FPS = FRAMES_PER_SECOND
    MOMENTUM = 0.01
    MIDFEAT = 1792

    # Training
    SEGMENT_SECONDS = 10.0
    BATCH_SIZE = 4
    NUM_EPOCHS = 3
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-4
    PATIENCE = 8

    # Freeze
    FREEZE_EPOCHS = 0
    FREEZE_CONV_BLOCKS = 0

config = Config()
