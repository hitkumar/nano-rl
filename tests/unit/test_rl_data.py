"""Unit tests for RL DataLoader"""

from unittest.mock import Mock

import pytest

from nano_rl.trainer.rl.data import DataLoader
from nano_rl.transport.types import MicroBatch


class TestDataLoaderGetBatch:
    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        return tokenizer

    @pytest.fixture
    def mock_receiver(self):
        return Mock()

    @pytest.fixture
    def mock_world(self):
        world = Mock()
        world.is_master = False
        world.rank = 0
        world.world_size = 1
        return world

    @pytest.fixture
    def data_loader(self, mock_tokenizer, mock_receiver, mock_world, monkeypatch):
        monkeypatch.setattr("nano_rl.trainer.rl.data.get_world", lambda: mock_world)
        monkeypatch.setattr(
            "nano_rl.trainer.rl.data.setup_micro_batch_receiver",
            lambda *args: mock_receiver,
        )

        loader = DataLoader(
            output_dir="/tmp/test",
            start_step=0,
            seq_len=128,
            tokenizer=mock_tokenizer,
            config=Mock(),
            dp_world_size=1,
        )
        return loader

    def test_get_batch_returns_list(self, data_loader, mock_receiver):
        """Test that get_batch returns a list of TensorBatch"""
        micro_batch = MicroBatch(
            input_ids=[1, 2, 3, 4],
            position_ids=[0, 1, 2, 3],
            loss_mask=[0, 0, 1, 1],
            advantages=[0.0, 0.0, 1.0, 1.0],
            inference_logprobs=[0.0, 0.0, -0.5, -0.3],
            temperature=1.0,
            ckpt_step=0,
        )
        mock_receiver.receive.return_value = [micro_batch]

        batches = data_loader.get_batch()

        assert isinstance(batches, list)
        assert len(batches) == 1

    def test_get_batch_converts_to_tensors(self, data_loader, mock_receiver):
        """Test that MicroBatch is correctly converted to TensorBatch"""
        micro_batch = MicroBatch(
            input_ids=[1, 2, 3, 4],
            position_ids=[0, 1, 2, 3],
            loss_mask=[0, 0, 1, 1],
            advantages=[0.0, 0.0, 1.0, 1.0],
            inference_logprobs=[0.0, 0.0, -0.5, -0.3],
            temperature=1.0,
            ckpt_step=5,
        )
        mock_receiver.receive.return_value = [micro_batch]

        batches = data_loader.get_batch()
        batch = batches[0]

        # Check shape is (1, seq_len) due to unsqueeze(0)
        assert batch["input_ids"].shape == (1, 4)
        assert batch["position_ids"].shape == (1, 4)
        assert batch["loss_mask"].shape == (1, 4)
        assert batch["advantages"].shape == (1, 4)
        assert batch["inference_logprobs"].shape == (1, 4)

        # Check values
        assert batch["input_ids"][0].tolist() == [1, 2, 3, 4]
        assert batch["position_ids"][0].tolist() == [0, 1, 2, 3]
        assert batch["loss_mask"][0].tolist() == [False, False, True, True]
        assert batch["advantages"][0].tolist() == [0.0, 0.0, 1.0, 1.0]
        # Use pytest.approx for float comparison due to float32 precision
        assert batch["inference_logprobs"][0].tolist() == pytest.approx(
            [0.0, 0.0, -0.5, -0.3]
        )

    def test_get_batch_includes_ckpt_step(self, data_loader, mock_receiver):
        """Test that ckpt_step is correctly passed through to TensorBatch"""
        micro_batch = MicroBatch(
            input_ids=[1, 2, 3, 4],
            position_ids=[0, 1, 2, 3],
            loss_mask=[0, 0, 1, 1],
            advantages=[0.0, 0.0, 1.0, 1.0],
            inference_logprobs=[0.0, 0.0, -0.5, -0.3],
            temperature=1.0,
            ckpt_step=3,
        )
        mock_receiver.receive.return_value = [micro_batch]

        batches = data_loader.get_batch()

        assert batches[0]["ckpt_step"] == 3

    def test_get_batch_includes_temperature(self, data_loader, mock_receiver):
        """Test that temperature is correctly passed through to TensorBatch"""
        micro_batch = MicroBatch(
            input_ids=[1, 2, 3, 4],
            position_ids=[0, 1, 2, 3],
            loss_mask=[0, 0, 1, 1],
            advantages=[0.0, 0.0, 1.0, 1.0],
            inference_logprobs=[0.0, 0.0, -0.5, -0.3],
            temperature=0.7,
            ckpt_step=0,
        )
        mock_receiver.receive.return_value = [micro_batch]

        batches = data_loader.get_batch()

        assert batches[0]["temperature"] == 0.7

    def test_get_batch_multiple_micro_batches(self, data_loader, mock_receiver):
        """Test that multiple MicroBatches are all converted"""
        micro_batch_1 = MicroBatch(
            input_ids=[1, 2],
            position_ids=[0, 1],
            loss_mask=[0, 1],
            advantages=[0.0, 1.0],
            inference_logprobs=[0.0, -0.5],
            temperature=1.0,
            ckpt_step=0,
        )
        micro_batch_2 = MicroBatch(
            input_ids=[3, 4],
            position_ids=[0, 1],
            loss_mask=[0, 1],
            advantages=[0.0, 2.0],
            inference_logprobs=[0.0, -0.3],
            temperature=1.0,
            ckpt_step=0,
        )
        mock_receiver.receive.return_value = [micro_batch_1, micro_batch_2]

        batches = data_loader.get_batch()

        assert len(batches) == 2
        assert batches[0]["input_ids"][0].tolist() == [1, 2]
        assert batches[1]["input_ids"][0].tolist() == [3, 4]
