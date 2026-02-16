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


def test_loss_scale_token_level():
    """When loss_scale is provided, loss is normalized by loss_scale instead of per-micro-batch token count."""
    inputs = _make_loss_inputs()

    # Without loss_scale: normalized by keep_mask.sum() internally
    loss_default, _ = compute_loss(**inputs, loss_config=LossConfig())

    # With loss_scale: normalized by loss_scale instead
    # Use the same value as internal normalization to get the same result
    loss_mask = inputs["loss_mask"]
    loss_config = LossConfig()
    log_ratio = inputs["trainer_logprobs"] - inputs["inference_logprobs"]
    ratio = torch.exp(log_ratio)
    tokens_masked = (ratio < loss_config.token_clip_low) | (ratio > loss_config.token_clip_high)
    keep_mask = loss_mask & ~tokens_masked
    num_keep = keep_mask.sum().item()

    loss_same, _ = compute_loss(**inputs, loss_config=loss_config, loss_scale=num_keep)
    assert abs(loss_same.item() - loss_default.item()) < 1e-5

    # With 2x loss_scale, loss should be half (since the sum is the same, just divided by 2x)
    loss_double, _ = compute_loss(**inputs, loss_config=loss_config, loss_scale=num_keep * 2)
    assert abs(loss_double.item() - loss_default.item() / 2) < 1e-5


def test_loss_scale_none_uses_micro_batch_normalization():
    """When loss_scale is None, loss is normalized by unmasked tokens in the micro-batch (default behavior)."""
    inputs = _make_loss_inputs()
    loss_none, _ = compute_loss(**inputs, loss_config=LossConfig(), loss_scale=None)
    loss_default, _ = compute_loss(**inputs, loss_config=LossConfig())
    assert loss_none.item() == loss_default.item()
