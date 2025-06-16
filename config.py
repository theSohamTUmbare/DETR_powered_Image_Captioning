import torch
from dataclasses import dataclass, field

@dataclass
class Config:
    VOCAB_SIZE: int = 49409
    CONTEXT_LEN: int = 64
    LEARNING_RATE: float = 1e-3
    BATCH_SIZE: int = 16
    DEVICE: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    MODEL_SAVE_PATH: str = "/kaggle/working"
    BASE_DIR: str = '/kaggle/input/coco-2017-dataset/coco2017'
    IDLE_DEVICE: str = 'cpu'
    PAD_ID = 49408
    LR = 1e-4
    MAX_EPOCHS: int = 1000


config = Config()