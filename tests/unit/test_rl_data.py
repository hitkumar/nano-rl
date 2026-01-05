"""Unit tests for RL DataLoader"""

from unittest.mock import MagicMock, Mock

import pytest
import torch

from nano_rl.trainer.rl.data import DataLoader, TensorBatch
from nano_rl.transport.types import TrainingBatch, TrainingSample


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
    def data_loader(self, mock_tokenizer, mock_receiver, monkeypatch):
        # Mock get_world and setup_training_batch_receiver
        monkeypatch.setattr("nano_rl.trainer.rl.data.get_world", lambda: Mock())
        monkeypatch.setattr(
            "nano_rl.trainer.rl.data.setup_training_batch_receiver",
            lambda *args: mock_receiver,
        )

        loader = DataLoader(
            output_dir="/tmp/test",
            start_step=0,
            seq_len=128,
            tokenizer=mock_tokenizer,
            config=Mock(),
        )
        return loader

    def test_get_batch_pads_to_max_len(
        self, data_loader, mock_receiver, mock_tokenizer
    ):
        """Test that shorter sequences are padded"""
        training_batch = TrainingBatch(
            examples=[
                TrainingSample(
                    prompt_ids=[1, 2],
                    prompt_mask=[True, True],
                    completion_ids=[3, 4, 5, 6],
                    completion_mask=[True, True, True, True],
                    completion_logprobs=[-0.5, -0.3, -0.4, -0.2],
                    advantage=1.0,
                ),
                TrainingSample(
                    prompt_ids=[1],
                    prompt_mask=[True],
                    completion_ids=[2],
                    completion_mask=[True],
                    completion_logprobs=[-0.5],
                    advantage=0.5,
                ),
            ],
            temperature=1.0,
            step=0,
        )
        mock_receiver.receive.return_value = training_batch

        batch = data_loader.get_batch()

        # Max len is 6 (from first example: 2 + 4)
        assert batch["input_ids"].shape == (2, 6)

        # Second example should be padded
        pad_id = mock_tokenizer.pad_token_id
        assert batch["input_ids"][1, 2:].tolist() == [pad_id] * 4

    def test_get_batch_truncates_to_seq_len(self, data_loader, mock_receiver):
        """Test that sequences are truncated to seq_len"""
        # Create example longer than seq_len (128)
        long_prompt = list(range(100))
        long_completion = list(range(100, 150))  # Total: 150 > 128

        training_batch = TrainingBatch(
            examples=[
                TrainingSample(
                    prompt_ids=long_prompt,
                    prompt_mask=[True] * 100,
                    completion_ids=long_completion,
                    completion_mask=[True] * 50,
                    completion_logprobs=[-0.5] * 50,
                    advantage=1.0,
                )
            ],
            temperature=1.0,
            step=0,
        )
        mock_receiver.receive.return_value = training_batch

        batch = data_loader.get_batch()

        # Should be truncated to seq_len
        assert batch["input_ids"].shape == (1, 128)

    def test_get_batch_none_advantage(self, data_loader, mock_receiver):
        """Test that None advantage defaults to 0.0"""
        training_batch = TrainingBatch(
            examples=[
                TrainingSample(
                    prompt_ids=[1, 2],
                    prompt_mask=[True, True],
                    completion_ids=[3, 4],
                    completion_mask=[True, True],
                    completion_logprobs=[-0.5, -0.3],
                    advantage=None,
                )
            ],
            temperature=1.0,
            step=0,
        )
        mock_receiver.receive.return_value = training_batch

        batch = data_loader.get_batch()

        # Advantages should be 0.0 for completion tokens
        assert batch["advantages"][0, 2:4].tolist() == [0.0, 0.0]
