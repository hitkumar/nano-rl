from abc import ABC, abstractmethod
from pathlib import Path

import msgspec
from nano_rl.transport.types import TrainingBatch


class TrainingBatchSender(ABC):
    """Base class for sending training batches from orchestrator to trainer"""

    def __init__(self, output_dir: Path):
        # self.logger = get_logger()
        self.encoder = msgspec.msgpack.Encoder()
        self.output_dir = output_dir

    @abstractmethod
    def send(self, batch: TrainingBatch) -> str:
        """Returns the path where the batch was saved"""
        pass

    def close(self) -> None:
        pass


class TrainingBatchReceiver(ABC):
    """Base class for receiving training batches"""

    def __init__(self):
        self.decoder = msgspec.msgpack.Decoder(type=TrainingBatch)

    @abstractmethod
    def wait(self) -> None:
        pass

    @abstractmethod
    def can_receive(self) -> bool:
        pass

    @abstractmethod
    def receive(self) -> TrainingBatch:
        pass

    def close(self) -> None:
        pass
