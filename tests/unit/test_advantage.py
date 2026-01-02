""" Unit tests for GRPO advantage computation """

import pytest
from nano_rl.orchestrator.advantage import compute_advantages
from nano_rl.orchestrator.config import AdvantageConfig


class TestComputeAdvantages:
    def test_no_config_returns_rewards(self):
        """When advantage_config is None, return rewards unchanged"""
        rewards = [1.0, 0.0, 0.5]
        result = compute_advantages(
            rewards=rewards,
            completion_lens=[10, 20, 30],
            rollouts_per_example=3,
            advantage_config=None,
        )
        assert result == rewards

    def test_uniform_rewards_zero_advantage(self):
        """When all rewards are the same, advantages should be 0"""
        rewards = [1.0, 1.0, 1.0]
        result = compute_advantages(
            rewards=rewards,
            completion_lens=[10, 10, 10],
            rollouts_per_example=3,
            advantage_config=AdvantageConfig(),
        )
        # All advantages should be 0 (or very close due to epsilon)
        for adv in result:
            assert abs(adv) < 1e-6

    def test_multiple_groups(self):
        """Test that normalization happens within each group independently"""
        # Group 1: [0, 1] -> different advantages
        # Group 2: [1, 1] -> same rewards, zero advantages
        rewards = [0.0, 1.0, 1.0, 1.0]
        result = compute_advantages(
            rewards=rewards,
            completion_lens=[10, 10, 10, 10],
            rollouts_per_example=2,
            advantage_config=AdvantageConfig(),
        )
        assert len(result) == 4
        # Group 1: should have non-zero advantages
        assert result[0] < 0  # low reward
        assert result[1] > 0  # high reward
        # Group 2: uniform rewards -> zero advantages
        assert abs(result[2]) < 1e-6
        assert abs(result[3]) < 1e-6

    def test_length_weighted_mean(self):
        """Test length-weighted mean baseline"""
        # Reward 1.0 with length 10, reward 0.0 with length 90
        # Normal mean: 0.5
        # Length-weighted mean: (1.0*10 + 0.0*90) / 100 = 0.1
        rewards = [1.0, 0.0]
        lens = [10, 90]
        config = AdvantageConfig(length_weighted_mean=True)
        result = compute_advantages(
            rewards=rewards,
            completion_lens=lens,
            rollouts_per_example=2,
            advantage_config=config,
        )
        # With length-weighted mean=0.1, the first reward (1.0) is far above mean
        # so its advantage should be larger than with normal mean
        assert result[0] > 0

    def test_length_weighted_adv(self):
        """Test that length_weighted_adv divides advantage by completion length"""
        rewards = [0.0, 1.0]
        config_no_weight = AdvantageConfig(length_weighted_adv=False)
        config_with_weight = AdvantageConfig(length_weighted_adv=True)

        result_no_weight = compute_advantages(
            rewards=rewards,
            completion_lens=[10, 100],
            rollouts_per_example=2,
            advantage_config=config_no_weight,
        )
        result_with_weight = compute_advantages(
            rewards=rewards,
            completion_lens=[10, 100],
            rollouts_per_example=2,
            advantage_config=config_with_weight,
        )
        # Second rollout has length 100, so its advantage should be scaled down
        assert abs(result_with_weight[1]) < abs(result_no_weight[1])

    def test_assertion_on_mismatched_lengths(self):
        """Test that assertion fails when rewards and completion_lens differ"""
        with pytest.raises(AssertionError):
            compute_advantages(
                rewards=[1.0, 0.0],
                completion_lens=[10],  # mismatched length
                rollouts_per_example=2,
                advantage_config=AdvantageConfig(),
            )

    def test_assertion_on_invalid_group_size(self):
        """Test that assertion fails when rewards don't divide evenly"""
        with pytest.raises(AssertionError):
            compute_advantages(
                rewards=[1.0, 0.0, 0.5],  # 3 rewards
                completion_lens=[10, 10, 10],
                rollouts_per_example=2,  # doesn't divide 3
                advantage_config=AdvantageConfig(),
            )
