""" Integration tests for filesystem transport """

from pathlib import Path

import pytest
from nano_rl.transport import (
    setup_training_batch_receiver,
    setup_training_batch_sender,
    TrainingBatch,
    TrainingSample,
)


def make_sample() -> TrainingSample:
    """Create a sample for testing."""
    return TrainingSample(
        prompt_ids=[101, 2054, 2003],
        prompt_mask=[True, True, True],
        completion_ids=[1996, 3437],
        completion_mask=[True, True],
        completion_logprobs=[-0.5, -1.2],
        advantage=0.75,
    )


def make_batch(step: int, ckpt_step: int = 0) -> TrainingBatch:
    return TrainingBatch(
        examples=[make_sample(), make_sample()],
        temperature=0.7,
        step=step,
        ckpt_step=ckpt_step,
    )


class TestFileSystemTransport:
    def test_send_receive_roundtrip(self, tmp_path: Path):
        sender = setup_training_batch_sender(tmp_path)
        receiver = setup_training_batch_receiver(tmp_path, current_step=0)
        batch = make_batch(step=0, ckpt_step=0)
        assert not receiver.can_receive()

        sender.send(batch)
        assert receiver.can_receive()
        received = receiver.receive()

        assert received.step == batch.step
        assert received.ckpt_step == batch.ckpt_step
        assert received.temperature == batch.temperature
        assert len(received.examples) == len(batch.examples)
        assert received.examples[0].prompt_ids == batch.examples[0].prompt_ids
        assert received.examples[0].advantage == batch.examples[0].advantage
        assert receiver.current_step == batch.step + 1

    def test_ckpt_step_preserved(self, tmp_path: Path):
        """Test that ckpt_step is correctly preserved through send/receive"""
        sender = setup_training_batch_sender(tmp_path)
        receiver = setup_training_batch_receiver(tmp_path, current_step=0)
        batch = make_batch(step=0, ckpt_step=5)

        sender.send(batch)
        received = receiver.receive()

        assert received.ckpt_step == 5
