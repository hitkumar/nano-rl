"""
RL Trainer config
"""

from pathlib import Path
from typing import Annotated, Literal

from nano_rl.trainer.config import (
    AdamWConfig,
    CheckpointConfig,
    ConstantSchedulerConfig,
    ModelConfig,
    OptimizerConfigType,
    SchedulerConfigType,
    TokenizerConfig,
)

from nano_rl.transport.config import FileSystemTransportConfig, TransportConfigType
from nano_rl.utils.config import LogConfig
from nano_rl.utils.pydantic_config import BaseConfig, BaseSettings
from pydantic import Field, model_validator


class LossConfig(BaseConfig):
    """Config for GRPO loss computation"""

    ratio_type: Annotated[
        Literal["token", "sequence"],
        Field(description="Type of importance ratio normalization."),
    ] = "token"

    # High threshold for clipping importance ratio
    # If π_new/π_old > 8.0, clip to 8.0 to prevent exploding gradients
    # This is similar to PPO clipping but applied to the ratio itself
    token_clip_high: Annotated[
        float,
        Field(
            ge=1.0, description="High threshold for token importance ratio clipping."
        ),
    ] = 8.0

    # Low threshold for clipping importance ratio
    # If π_new/π_old < 0.125, clip to 0.125
    # Prevents the policy from moving too far from the old policy
    token_clip_low: Annotated[
        float,
        Field(
            ge=0,
            le=1.0,
            description="Low threshold for token importance ratio clipping.",
        ),
    ] = 0.125

    # KL divergence penalty coefficient
    # If > 0, adds a penalty term to prevent policy from diverging too much
    # Loss += kl_coef * KL(π_new || π_old)
    kl_coef: Annotated[
        float,
        Field(ge=0, description="Coefficient for KL penalty term."),
    ] = 0.0

    @model_validator(mode="after")
    def validate_mask_bounds(self):
        if self.token_clip_low >= self.token_clip_high:
            raise ValueError(
                f"token_mask_low ({self.token_clip_low}) must be less than token_mask_high ({self.token_clip_high})"
            )
        return self


class FakeDataLoaderConfig(BaseConfig):
    """Configures a fake data loader sampling random micro batches for debugging."""

    batch_size: Annotated[int, Field(ge=1)] = 2
    generate_samples: Annotated[
        bool,
        Field(
            description="Whether to generate separate samples and pack them into a single micro batch."
        ),
    ] = False


class DataLoaderConfig(BaseConfig):
    """Configures the data loader used for training."""

    fake: Annotated[
        FakeDataLoaderConfig | None,
        Field(description="Whether to use a fake data loader."),
    ] = None


class RlTrainerConfig(BaseSettings):
    """Config for the RL trainer"""

    # Configures the model to train.
    model: ModelConfig = ModelConfig()

    # Configures the tokenizer to use.
    tokenizer: TokenizerConfig = TokenizerConfig()

    # Data loading config.
    data: DataLoaderConfig = DataLoaderConfig()

    # Configures the optimizer to use.
    optim: Annotated[OptimizerConfigType, Field(discriminator="type")] = AdamWConfig()

    # Configures the scheduler to use.
    scheduler: Annotated[SchedulerConfigType, Field(discriminator="type")] = (
        ConstantSchedulerConfig()
    )

    # Configures the loss to use.
    loss: LossConfig = LossConfig()

    # ckpt config
    ckpt: CheckpointConfig = CheckpointConfig()

    # Transport mechanism for receiving rollouts from orchestrator
    # Currently only filesystem transport is supported
    rollout_transport: Annotated[TransportConfigType, Field(discriminator="type")] = (
        FileSystemTransportConfig()
    )

    # Logging configuration
    log: LogConfig = LogConfig()

    # Output directory - MUST match orchestrator's output_dir
    # This is where rollouts are written and weights are saved
    output_dir: Annotated[
        Path,
        Field(
            description="Directory for outputs (checkpoints, weights, logs). Must match orchestrator's output_dir."
        ),
    ] = Path("outputs")

    # Maximum training steps
    # If None, trains indefinitely (useful for continuous RL)
    max_steps: Annotated[
        int | None,
        Field(description="Maximum training steps. If None, runs indefinitely."),
    ] = None

    dist_timeout_seconds: Annotated[
        int,
        Field(
            description="Timeout in seconds for torch distributed ops. Defaults to 600 seconds.",
        ),
    ] = 600

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        if self.tokenizer.name is None:
            self.tokenizer.name = self.model.name
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.model.trust_remote_code

        return self
