"""Unit tests for RL Packer"""

from unittest.mock import Mock

import pytest

from nano_rl.trainer.rl.packer import (
    pack_samples_into_micro_batches,
    prepare_batch,
    prepare_sample,
)
from nano_rl.transport.types import MicroBatch, TrainingBatch, TrainingSample


class TestPrepareSample:
    def test_concatenates_prompt_and_completion(self):
        """Test that prompt and completion ids are concatenated"""
        sample = TrainingSample(
            prompt_ids=[1, 2, 3],
            prompt_mask=[True, True, True],
            completion_ids=[4, 5],
            completion_mask=[True, True],
            completion_logprobs=[-0.5, -0.3],
            advantage=1.0,
        )

        micro_batch = prepare_sample(sample, temperature=0.7, ckpt_step=5)

        assert micro_batch.input_ids == [1, 2, 3, 4, 5]
        assert micro_batch.position_ids == [0, 1, 2, 3, 4]

    def test_loss_mask_zeros_for_prompt(self):
        """Test that loss_mask is 0 for prompt tokens and 1 for completion"""
        sample = TrainingSample(
            prompt_ids=[1, 2],
            prompt_mask=[True, True],
            completion_ids=[3, 4, 5],
            completion_mask=[True, True, True],
            completion_logprobs=[-0.5, -0.3, -0.2],
            advantage=1.0,
        )

        micro_batch = prepare_sample(sample, temperature=0.7, ckpt_step=0)

        assert micro_batch.loss_mask == [0, 0, 1, 1, 1]

    def test_advantages_zeros_for_prompt(self):
        """Test that advantages are 0 for prompt and advantage value for completion"""
        sample = TrainingSample(
            prompt_ids=[1, 2],
            prompt_mask=[True, True],
            completion_ids=[3, 4],
            completion_mask=[True, True],
            completion_logprobs=[-0.5, -0.3],
            advantage=2.5,
        )

        micro_batch = prepare_sample(sample, temperature=0.7, ckpt_step=0)

        assert micro_batch.advantages == [0, 0, 2.5, 2.5]

    def test_none_advantage_defaults_to_zero(self):
        """Test that None advantage is treated as 0.0"""
        sample = TrainingSample(
            prompt_ids=[1],
            prompt_mask=[True],
            completion_ids=[2, 3],
            completion_mask=[True, True],
            completion_logprobs=[-0.5, -0.3],
            advantage=None,
        )

        micro_batch = prepare_sample(sample, temperature=0.7, ckpt_step=0)

        assert micro_batch.advantages == [0, 0.0, 0.0]

    def test_inference_logprobs_zeros_for_prompt(self):
        """Test that inference_logprobs are 0 for prompt"""
        sample = TrainingSample(
            prompt_ids=[1, 2],
            prompt_mask=[True, True],
            completion_ids=[3, 4],
            completion_mask=[True, True],
            completion_logprobs=[-0.5, -0.3],
            advantage=1.0,
        )

        micro_batch = prepare_sample(sample, temperature=0.7, ckpt_step=0)

        assert micro_batch.inference_logprobs == [0, 0, -0.5, -0.3]

    def test_temperature_and_ckpt_step_passed_through(self):
        """Test that temperature and ckpt_step are set correctly"""
        sample = TrainingSample(
            prompt_ids=[1],
            prompt_mask=[True],
            completion_ids=[2],
            completion_mask=[True],
            completion_logprobs=[-0.5],
            advantage=1.0,
        )

        micro_batch = prepare_sample(sample, temperature=0.8, ckpt_step=42)

        assert micro_batch.temperature == 0.8
        assert micro_batch.ckpt_step == 42


class TestPackSamplesIntoMicroBatches:
    def test_empty_samples_returns_empty(self):
        """Test that empty input returns empty output"""
        result = pack_samples_into_micro_batches([], seq_len=10, pad_int=0)
        assert result == []

    def test_single_sample_padded_to_seq_len(self):
        """Test that a single sample is padded to seq_len"""
        sample = MicroBatch(
            input_ids=[1, 2, 3],
            position_ids=[0, 1, 2],
            loss_mask=[0, 1, 1],
            advantages=[0.0, 1.0, 1.0],
            inference_logprobs=[0.0, -0.5, -0.3],
            temperature=1.0,
            ckpt_step=0,
        )

        result = pack_samples_into_micro_batches([sample], seq_len=5, pad_int=0)

        assert len(result) == 1
        assert len(result[0].input_ids) == 5
        assert result[0].input_ids == [1, 2, 3, 0, 0]
        assert result[0].loss_mask == [0, 1, 1, 0, 0]

    def test_samples_packed_together(self):
        """Test that two small samples are packed into one micro batch"""
        sample1 = MicroBatch(
            input_ids=[1, 2],
            position_ids=[0, 1],
            loss_mask=[0, 1],
            advantages=[0.0, 1.0],
            inference_logprobs=[0.0, -0.5],
            temperature=1.0,
            ckpt_step=0,
        )
        sample2 = MicroBatch(
            input_ids=[3, 4],
            position_ids=[0, 1],
            loss_mask=[0, 1],
            advantages=[0.0, 2.0],
            inference_logprobs=[0.0, -0.3],
            temperature=1.0,
            ckpt_step=0,
        )

        result = pack_samples_into_micro_batches([sample1, sample2], seq_len=6, pad_int=0)

        assert len(result) == 1
        assert len(result[0].input_ids) == 6
        # Samples are packed together with padding
        assert result[0].input_ids[:4] == [1, 2, 3, 4]

    def test_samples_too_large_not_packed(self):
        """Test that samples that don't fit together are in separate batches"""
        sample1 = MicroBatch(
            input_ids=[1, 2, 3],
            position_ids=[0, 1, 2],
            loss_mask=[0, 1, 1],
            advantages=[0.0, 1.0, 1.0],
            inference_logprobs=[0.0, -0.5, -0.3],
            temperature=1.0,
            ckpt_step=0,
        )
        sample2 = MicroBatch(
            input_ids=[4, 5, 6],
            position_ids=[0, 1, 2],
            loss_mask=[0, 1, 1],
            advantages=[0.0, 2.0, 2.0],
            inference_logprobs=[0.0, -0.4, -0.2],
            temperature=1.0,
            ckpt_step=0,
        )

        result = pack_samples_into_micro_batches([sample1, sample2], seq_len=4, pad_int=0)

        assert len(result) == 2

    def test_truncates_long_samples(self):
        """Test that samples longer than seq_len are truncated"""
        sample = MicroBatch(
            input_ids=[1, 2, 3, 4, 5],
            position_ids=[0, 1, 2, 3, 4],
            loss_mask=[0, 1, 1, 1, 1],
            advantages=[0.0, 1.0, 1.0, 1.0, 1.0],
            inference_logprobs=[0.0, -0.5, -0.3, -0.2, -0.1],
            temperature=1.0,
            ckpt_step=0,
        )

        result = pack_samples_into_micro_batches([sample], seq_len=3, pad_int=0)

        assert len(result) == 1
        assert len(result[0].input_ids) == 3
        assert result[0].input_ids == [1, 2, 3]


class TestPrepareBatch:
    @pytest.fixture
    def mock_logger(self, monkeypatch):
        logger = Mock()
        monkeypatch.setattr(
            "nano_rl.trainer.rl.packer.get_logger", lambda: logger
        )
        return logger

    def test_creates_grid_for_dp_ranks(self, mock_logger):
        """Test that prepare_batch creates correct grid for DP ranks"""
        training_batch = TrainingBatch(
            examples=[
                TrainingSample(
                    prompt_ids=[1, 2],
                    prompt_mask=[True, True],
                    completion_ids=[3, 4],
                    completion_mask=[True, True],
                    completion_logprobs=[-0.5, -0.3],
                    advantage=1.0,
                ),
                TrainingSample(
                    prompt_ids=[5, 6],
                    prompt_mask=[True, True],
                    completion_ids=[7, 8],
                    completion_mask=[True, True],
                    completion_logprobs=[-0.4, -0.2],
                    advantage=0.5,
                ),
            ],
            temperature=0.7,
            step=0,
            ckpt_step=5,
        )

        grid = prepare_batch(training_batch, dp_world_size=2, seq_len=8, pad_id=0)

        assert len(grid) == 2
        # Each rank should have at least one batch
        assert len(grid[0]) >= 1
        assert len(grid[1]) >= 1

    def test_pads_to_divisible_by_dp_world_size(self, mock_logger):
        """Test that dummy batches are added when not divisible"""
        training_batch = TrainingBatch(
            examples=[
                TrainingSample(
                    prompt_ids=[1, 2, 3, 4],
                    prompt_mask=[True, True, True, True],
                    completion_ids=[5, 6, 7, 8],
                    completion_mask=[True, True, True, True],
                    completion_logprobs=[-0.5, -0.3, -0.2, -0.1],
                    advantage=1.0,
                ),
            ],
            temperature=0.7,
            step=0,
            ckpt_step=5,
        )

        # 1 sample that can't be packed into fewer than 1 batch
        # With dp_world_size=2, we need 2 batches (1 real + 1 dummy)
        grid = prepare_batch(training_batch, dp_world_size=2, seq_len=10, pad_id=0)

        assert len(grid) == 2
        total_batches = sum(len(rank_batches) for rank_batches in grid)
        assert total_batches % 2 == 0

    def test_logs_warning_when_padding_needed(self, mock_logger):
        """Test that a warning is logged when dummy batches are needed"""
        training_batch = TrainingBatch(
            examples=[
                TrainingSample(
                    prompt_ids=[1, 2, 3, 4, 5],
                    prompt_mask=[True] * 5,
                    completion_ids=[6, 7, 8, 9, 10],
                    completion_mask=[True] * 5,
                    completion_logprobs=[-0.5] * 5,
                    advantage=1.0,
                ),
            ],
            temperature=0.7,
            step=0,
            ckpt_step=5,
        )

        # 1 batch with dp_world_size=2 requires padding
        prepare_batch(training_batch, dp_world_size=2, seq_len=12, pad_id=0)

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        assert "not divisible" in call_args

    def test_all_batches_have_correct_seq_len(self, mock_logger):
        """Test that all output batches have correct sequence length"""
        training_batch = TrainingBatch(
            examples=[
                TrainingSample(
                    prompt_ids=[1, 2],
                    prompt_mask=[True, True],
                    completion_ids=[3],
                    completion_mask=[True],
                    completion_logprobs=[-0.5],
                    advantage=1.0,
                ),
            ],
            temperature=0.7,
            step=0,
            ckpt_step=5,
        )

        grid = prepare_batch(training_batch, dp_world_size=1, seq_len=8, pad_id=0)

        for rank_batches in grid:
            for batch in rank_batches:
                assert len(batch.input_ids) == 8
                assert len(batch.position_ids) == 8
                assert len(batch.loss_mask) == 8
                assert len(batch.advantages) == 8
                assert len(batch.inference_logprobs) == 8
