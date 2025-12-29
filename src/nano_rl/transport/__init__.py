""" Transport layer for orchestrator-trainer communication"""

from pathlib import Path

from multiprocess.sharedctypes import Value

from nano_rl.transport.base import TrainingBatchReceiver, TrainingBatchSender
from nano_rl.transport.config import FileSystemTransportConfig, TransportConfigType
from nano_rl.transport.filesystem import (
    FileSystemTrainingBatchReceiver,
    FileSystemTrainingBatchSender,
)
from nano_rl.transport.types import TrainingBatch, TrainingSample


def setup_training_batch_sender(
    output_dir: Path, transport_type: TransportConfigType = FileSystemTransportConfig()
) -> TrainingBatchSender:
    if transport_type.type == "filesystem":
        return FileSystemTrainingBatchSender(output_dir)

    raise ValueError(f"Invalid transport type passed: {transport_type.type}")


def setup_training_batch_receiver(
    output_dir: Path,
    current_step: int,
    transport_type: TransportConfigType = FileSystemTransportConfig(),
) -> TrainingBatchReceiver:
    if transport_type.type == "filesystem":
        return FileSystemTrainingBatchReceiver(output_dir, current_step)

    raise ValueError(f"Invalid transport type passed: {transport_type.type}")


# modules imported by wildcard import.
__all__ = [
    "TrainingSample",
    "TrainingBatch",
    "TrainingBatchReceiver",
    "TrainingBatchSender",
    "TransportConfigType",
    "setup_training_batch_sender",
    "setup_training_batch_receiver",
]
