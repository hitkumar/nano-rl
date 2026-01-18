"""
Validation utilities for ensuring config consistency across components.
"""

from nano_rl.orchestrator.config import OrchestratorConfig
from nano_rl.trainer.rl.config import RlTrainerConfig


def validate_shared_max_async_level(
    trainer: RlTrainerConfig,
    orchestrator: OrchestratorConfig,
) -> None:
    """
    Validate that the max_async_level is the same for trainer and orchestrator.
    """
    if trainer.max_async_level != orchestrator.max_async_level:
        raise ValueError(
            f"Trainer max_async_level ({trainer.max_async_level}) and orchestrator "
            f"max_async_level ({orchestrator.max_async_level}) must be the same."
        )
