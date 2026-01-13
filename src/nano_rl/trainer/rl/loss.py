"""
RL loss computation for GRPO training
"""

import torch
import torch.nn.functional as F
from beartype import beartype as typechecker
from jaxtyping import Bool, Float, Int, jaxtyped

from nano_rl.trainer.rl.config import LossConfig
from torch import Tensor


def _safe_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of values over a bool mask, returns 0 if mask is empty"""
    denom = torch.clamp_min(mask.sum(), min=1)
    return values[mask].sum() / denom


@jaxtyped(typechecker=typechecker)
def shift_logits(
    logits: Float[Tensor, "batch seq vocab"],
    left_pad_logit: Float[Tensor, "batch 1 vocab"] | None = None,
) -> Float[Tensor, "batch seq vocab"]:
    """Removes last logit as we don't have the target for it and aligns logits with input seq, so logits[b, t] corresponds to input token (b, t)"""
    batch, seq, vocab = logits.shape
    shifted_logits = logits[:, :-1, :]
    if left_pad_logit is None:
        left_pad_logit = torch.zeros(
            batch, 1, vocab, device=logits.device, dtype=logits.dtype
        )
    shifted_logits = torch.cat(
        [left_pad_logit, shifted_logits], dim=1
    )  # (batch_size, seq_len, vocab_size)
    return shifted_logits


@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def compute_entropy(
    shifted_logits: Float[Tensor, "batch seq vocab"]
) -> Float[Tensor, "batch seq"]:
    """Used to keep track of policy entropy during training, not used during loss calc"""
    with torch.no_grad():
        pd = F.softmax(shifted_logits, dim=-1)
        entropy = torch.logsumexp(shifted_logits, dim=-1) - torch.sum(
            pd * shifted_logits, dim=-1
        )
    return entropy


@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def selective_log_softmax(
    logits: Float[Tensor, "batch seq vocab"], index: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq"]:
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


def compute_loss(
    trainer_logprobs: Float[Tensor, "batch seq"],
    inference_logprobs: Float[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],  # same for all tokens in the sequence,
    loss_mask: Bool[Tensor, "batch seq"],
    loss_config: LossConfig,
) -> tuple[Float[Tensor, ""], dict[str, Tensor]]:
    """Computes GRPO loss"""

    log_importance_ratio = trainer_logprobs - inference_logprobs
    importance_ratio = torch.exp(log_importance_ratio)

    # used to track how far apart the policies are, 0 when r=0 meaning policies are identical and always >= 0
    mismatch_kl = importance_ratio - log_importance_ratio - 1

    token_mask_low = importance_ratio < loss_config.token_clip_low
    token_mask_high = importance_ratio > loss_config.token_clip_high
    tokens_masked = token_mask_low | token_mask_high
    keep_mask = loss_mask & ~tokens_masked

    if loss_config.ratio_type == "sequence":
        seq_log_ratio = _safe_mean(log_importance_ratio, loss_mask)
        # geometric mean of ratios
        seq_ratio = torch.exp(torch.clamp(seq_log_ratio, max=10.0))
        coeff = seq_ratio * (advantages - loss_config.kl_coef * log_importance_ratio)
    else:
        coeff = importance_ratio * (
            advantages - loss_config.kl_coef * log_importance_ratio
        )

    # we detach coeff here so that gradient doesn't flow this, it is meant to be a constant for policy gradient
    # if coeff > 0, gradient increase log_probs, coeff < 0 gradient decreases log_probs
    loss = -(coeff.detach() * trainer_logprobs)[keep_mask].sum()
    num_tokens = keep_mask.sum().clamp(min=1)
    loss = loss / num_tokens
    diagnostics = {
        "mismatch_kl": _safe_mean(mismatch_kl, loss_mask).detach(),
        "tokens_masked": tokens_masked[loss_mask].float().mean().detach(),
        "is_masked_low": token_mask_low[loss_mask].float().mean().detach(),
        "is_masked_high": token_mask_high[loss_mask].float().mean().detach(),
        "importance_ratio": _safe_mean(importance_ratio, loss_mask).detach(),
    }

    return loss, diagnostics
