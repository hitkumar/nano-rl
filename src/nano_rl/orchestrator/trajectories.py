"""
Trajectory interleaving: converts multi-turn rollouts into flat TrainingSamples.

When consecutive trajectory steps share token prefixes (the extension property),
they are merged into a single sample. Environment response tokens (feedback between
turns) are included with mask=False (no gradient). Model completion tokens keep
their actual masks and logprobs.

For single-turn rollouts, this produces one sample
"""

import verifiers as vf
from nano_rl.transport.types import TrainingSample
from nano_rl.utils.logger import get_logger


def make_sample(step: vf.TrajectoryStep, has_error: bool) -> TrainingSample:
    """Create a new TrainingSample from a single trajectory step."""
    tokens = step["tokens"]
    if has_error:
        completion_mask = [False] * len(tokens["completion_mask"])
    else:
        completion_mask = [bool(i) for i in tokens["completion_mask"]]

    return TrainingSample(
        prompt_ids=list(tokens["prompt_ids"]),
        prompt_mask=[bool(i) for i in tokens["prompt_mask"]],
        completion_ids=list(tokens["completion_ids"]),
        completion_mask=completion_mask,
        completion_logprobs=list(tokens["completion_logprobs"]),
    )


def extend_sample(
    sample: TrainingSample, step: vf.TrajectoryStep, prefix_len: int, has_error: bool
) -> None:
    """Extend an existing sample with a new trajectory step (extension property holds).

    New prompt tokens beyond the shared prefix are environment response tokens —
    appended with mask=False and logprobs=0.0. New completion tokens are the model's
    response — appended with their actual masks and logprobs.
    """
    tokens = step["tokens"]

    # Environment response tokens (no gradient)
    new_prompt_ids = tokens["prompt_ids"][prefix_len:]
    sample.completion_ids.extend(new_prompt_ids)
    sample.completion_mask.extend([False] * len(new_prompt_ids))
    sample.completion_logprobs.extend([0.0] * len(new_prompt_ids))

    # Model completion tokens
    sample.completion_ids.extend(tokens["completion_ids"])
    if has_error:
        sample.completion_mask.extend([False] * len(tokens["completion_mask"]))
    else:
        sample.completion_mask.extend(bool(i) for i in tokens["completion_mask"])
    sample.completion_logprobs.extend(tokens["completion_logprobs"])


def interleave_rollout(output: vf.RolloutOutput) -> list[TrainingSample] | None:
    """Convert a rollout output into one or more TrainingSamples.

    Walks all trajectory steps and merges consecutive steps that share token
    prefixes (extension property) into a single flat sample. When extension
    breaks, starts a new sample.

    Returns None if the trajectory is empty.
    """
    logger = get_logger()
    trajectory = output["trajectory"]

    if not trajectory:
        logger.warning("Empty trajectory, skipping rollout")
        return None

    has_error = output.get("error") is not None

    # Initialize with first step
    first_tokens = trajectory[0]["tokens"]
    first_prefix = first_tokens["prompt_ids"] + first_tokens["completion_ids"]
    active_samples: list[list] = [[first_prefix, make_sample(trajectory[0], has_error)]]

    for step_idx, step in enumerate(trajectory[1:], start=1):
        tokens = step["tokens"]
        step_prompt_ids = tokens["prompt_ids"]

        # Check if this step extends any active prefix
        matched_idx = None
        for idx, (prefix_tokens, _) in enumerate(active_samples):
            if step_prompt_ids[: len(prefix_tokens)] == prefix_tokens:
                matched_idx = idx
                break

        if matched_idx is not None:
            prefix_tokens, sample = active_samples[matched_idx]
            extend_sample(sample, step, len(prefix_tokens), has_error)
            active_samples[matched_idx][0] = (
                tokens["prompt_ids"] + tokens["completion_ids"]
            )
        else:
            logger.debug(
                f"Extension property broke at step {step_idx}, starting new sample"
            )
            new_prefix = tokens["prompt_ids"] + tokens["completion_ids"]
            active_samples.append([new_prefix, make_sample(step, has_error)])

    return [sample for _, sample in active_samples]
