from abc import ABC, abstractmethod
from pathlib import Path

import torch.nn as nn
from nano_rl.utils.logger import get_logger


class WeightBroadcast(ABC):
    def __init__(self, output_dir: Path):
        self.logger = get_logger()
        self.output_dir = output_dir

    @abstractmethod
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        pass
