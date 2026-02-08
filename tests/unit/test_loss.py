"""Unit tests for GRPO loss computation"""

import pytest
import torch

from nano_rl.trainer.rl.config import LossConfig
from nano_rl.trainer.rl.loss import compute_entropy, compute_loss, selective_log_softmax, shift_logits


def test_shift_logits():
    logits = torch.arange(12, dtype=torch.float).reshape(1, 4, 3)
    shifted = shift_logits(logits)
    assert (shifted[:, 0, :] == 0).all()
    assert torch.equal(shifted[:, 1:, :], logits[:, :-1, :])


def test_selective_log_softmax():
    logits = torch.randn(2, 5, 100)
    index = torch.randint(0, 100, (2, 5))
    result = selective_log_softmax(logits, index)
    assert result.shape == (2, 5)
    assert (result <= 0).all()


def test_compute_entropy():
    entropy = compute_entropy(torch.randn(2, 5, 100))
    assert entropy.shape == (2, 5)
    assert (entropy >= 0).all()


def _make_loss_inputs(batch=2, seq=10):
    return dict(
        trainer_logprobs=torch.randn(batch, seq),
        inference_logprobs=torch.randn(batch, seq),
        advantages=torch.randn(batch, seq),
        loss_mask=torch.ones(batch, seq, dtype=torch.bool),
    )


def test_token_ratio_loss():
    loss, diag = compute_loss(**_make_loss_inputs(), loss_config=LossConfig())
    assert loss.shape == ()
    assert "importance_ratio" in diag


def test_sequence_ratio_loss():
    loss, _ = compute_loss(**_make_loss_inputs(), loss_config=LossConfig(ratio_type="sequence"))
    assert loss.shape == ()


def test_on_policy_kl_is_zero():
    logprobs = torch.randn(2, 10)
    _, diag = compute_loss(
        trainer_logprobs=logprobs,
        inference_logprobs=logprobs,
        advantages=torch.zeros(2, 10),
        loss_mask=torch.ones(2, 10, dtype=torch.bool),
        loss_config=LossConfig(),
    )
    assert diag["mismatch_kl"].item() == pytest.approx(0.0, abs=1e-5)


def test_clipping_masks_extreme_ratios():
    _, diag = compute_loss(
        trainer_logprobs=torch.zeros(1, 10),
        inference_logprobs=torch.full((1, 10), -10.0),
        advantages=torch.ones(1, 10),
        loss_mask=torch.ones(1, 10, dtype=torch.bool),
        loss_config=LossConfig(token_clip_high=8.0),
    )
    assert diag["tokens_masked"].item() > 0


def test_empty_mask_returns_zero_loss():
    loss, _ = compute_loss(**_make_loss_inputs(), loss_config=LossConfig())
    loss_empty, _ = compute_loss(
        **{**_make_loss_inputs(), "loss_mask": torch.zeros(2, 10, dtype=torch.bool)},
        loss_config=LossConfig(),
    )
    assert loss_empty.item() == 0.0
