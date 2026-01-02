""" GRPO Advantage computation """

from nano_rl.orchestrator.config import AdvantageConfig


def compute_advantages(
    rewards: list[float],
    completion_lens: list[int],
    rollouts_per_example: int,
    advantage_config: AdvantageConfig | None,
) -> list[float]:
    """
    Computes GRPO advantages by normalizing rewards within each group
    """
    if not advantage_config:
        return rewards

    assert len(rewards) == len(completion_lens)
    assert (
        len(rewards) % rollouts_per_example == 0
    ), "Number of rewards should be a multiple of rollouts_per_example"
    num_groups = len(rewards) // rollouts_per_example
    advantages = []

    for g in range(num_groups):
        group_start_idx = g * rollouts_per_example
        group_end_idx = group_start_idx + rollouts_per_example
        group_rewards = rewards[group_start_idx:group_end_idx]
        group_completion_lens = completion_lens[group_start_idx:group_end_idx]

        if advantage_config.length_weighted_mean:
            group_mean = sum(
                r * l for (r, l) in zip(group_rewards, group_completion_lens)
            ) / sum(group_completion_lens)
        else:
            group_mean = sum(group_rewards) / len(group_rewards)

        # population variance, so we divide by N not N-1
        group_var = sum((r - group_mean) ** 2 for r in group_rewards) / len(
            group_rewards
        )
        group_std = group_var**0.5 if group_var > 0 else 1.0
        for i, r in enumerate(group_rewards):
            # prevent divide by 0 errors
            adv = (r - group_mean) / (group_std + 1e-8)
            if advantage_config.length_weighted_adv:
                adv = adv / group_completion_lens[i]

            advantages.append(adv)

    return advantages
