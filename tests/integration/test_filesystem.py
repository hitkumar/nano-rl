""" Integration tests for filesystem transport """

from pathlib import Path

import pytest
from nano_rl.transport import (
    MicroBatch,
    setup_micro_batch_receiver,
    setup_micro_batch_sender,
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


def make_micro_batch(ckpt_step: int = 0) -> MicroBatch:
    """Create a micro batch for testing."""
    return MicroBatch(
        input_ids=[1, 2, 3, 4, 5],
        position_ids=[0, 1, 2, 3, 4],
        loss_mask=[0, 0, 1, 1, 1],
        advantages=[0.0, 0.0, 0.5, 0.5, 0.5],
        inference_logprobs=[0.0, 0.0, -0.5, -0.3, -0.2],
        temperature=0.7,
        ckpt_step=ckpt_step,
    )


class TestFileSystemMicroBatchTransport:
    def test_send_receive_single_rank(self, tmp_path: Path):
        """Test sending and receiving micro batches for a single DP rank"""
        dp_world_size = 1
        sender = setup_micro_batch_sender(tmp_path, dp_world_size, start_step=0)
        receiver = setup_micro_batch_receiver(tmp_path, dp_rank=0, start_step=0)

        micro_batch = make_micro_batch(ckpt_step=3)
        micro_batch_grid = [[micro_batch]]

        sender.send(micro_batch_grid, step=0)
        received = receiver.receive()

        assert len(received) == 1
        assert received[0].input_ids == micro_batch.input_ids
        assert received[0].position_ids == micro_batch.position_ids
        assert received[0].loss_mask == micro_batch.loss_mask
        assert received[0].advantages == micro_batch.advantages
        assert received[0].inference_logprobs == micro_batch.inference_logprobs
        assert received[0].temperature == micro_batch.temperature
        assert received[0].ckpt_step == 3

    def test_send_receive_multiple_ranks(self, tmp_path: Path):
        """Test sending and receiving micro batches for multiple DP ranks"""
        dp_world_size = 2
        sender = setup_micro_batch_sender(tmp_path, dp_world_size, start_step=0)
        receiver_0 = setup_micro_batch_receiver(tmp_path, dp_rank=0, start_step=0)
        receiver_1 = setup_micro_batch_receiver(tmp_path, dp_rank=1, start_step=0)

        micro_batch_0 = make_micro_batch(ckpt_step=1)
        micro_batch_1 = make_micro_batch(ckpt_step=2)
        micro_batch_grid = [[micro_batch_0], [micro_batch_1]]

        sender.send(micro_batch_grid, step=0)

        received_0 = receiver_0.receive()
        received_1 = receiver_1.receive()

        assert len(received_0) == 1
        assert len(received_1) == 1
        assert received_0[0].ckpt_step == 1
        assert received_1[0].ckpt_step == 2

    def test_send_receive_multiple_batches_per_rank(self, tmp_path: Path):
        """Test sending multiple micro batches per DP rank"""
        dp_world_size = 1
        sender = setup_micro_batch_sender(tmp_path, dp_world_size, start_step=0)
        receiver = setup_micro_batch_receiver(tmp_path, dp_rank=0, start_step=0)

        micro_batch_1 = make_micro_batch(ckpt_step=1)
        micro_batch_2 = make_micro_batch(ckpt_step=2)
        micro_batch_grid = [[micro_batch_1, micro_batch_2]]

        sender.send(micro_batch_grid, step=0)
        received = receiver.receive()

        assert len(received) == 2
        assert received[0].ckpt_step == 1
        assert received[1].ckpt_step == 2

    def test_receiver_step_increments(self, tmp_path: Path):
        """Test that receiver step increments after receive"""
        dp_world_size = 1
        sender = setup_micro_batch_sender(tmp_path, dp_world_size, start_step=0)
        receiver = setup_micro_batch_receiver(tmp_path, dp_rank=0, start_step=0)

        micro_batch_grid = [[make_micro_batch()]]

        sender.send(micro_batch_grid, step=0)
        assert receiver.current_step == 0

        receiver.receive()
        assert receiver.current_step == 1

