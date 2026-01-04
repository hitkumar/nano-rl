"""
Orchestration config
"""

from pathlib import Path
from typing import Annotated, Any

from nano_rl.transport.config import FileSystemTransportConfig, TransportConfigType

from nano_rl.utils.config import ClientConfig, LogConfig, ModelConfig
from nano_rl.utils.pydantic_config import BaseConfig, BaseSettings
from pydantic import Field, model_validator


class SamplingConfig(BaseConfig):
    """Configures how tokens are sampled from the model for training. Largely follows the vLLM sampling parameters."""

    temperature: Annotated[
        float,
        Field(
            ge=0,
            description="Scales the output probability distribution. Lower values => more deterministic, higher values => more random. If 0, will sample greedily.",
        ),
    ] = 1.0

    repetition_penalty: Annotated[
        float,
        Field(
            ge=0,
            description="Penalty for repeating tokens. Values > 1.0 discourage repetition, values < 1.0 encourage repetition, and 1.0 means no penalty.",
        ),
    ] = 1.0

    max_tokens: Annotated[
        int | None,
        Field(
            description="Maximum number of output tokens to generate per turn. If None, will generate until maximum context length or EOS token is hit.",
        ),
    ] = None

    min_tokens: Annotated[
        int,
        Field(
            ge=0,
            description="Minimum number of output tokens to generate per sequence.",
        ),
    ] = 0

    seed: Annotated[
        int | None,
        Field(
            description="Random seed to use for sampling. If None, no seeding is used.",
        ),
    ] = None

    # Strictly speaking, extra_body is not a sampling parameter, but it is the
    # easiest way to pass arbitrary extra parameters to the server via verifiers
    extra_body: Annotated[
        dict[str, Any],
        Field(
            description="Extra body to pass with each request to the inference server. By default, it is set to an empty dictionary.",
        ),
    ] = {}
    top_p: Annotated[
        float | None,
        Field(
            description="Cumulative probability of the top tokens to consider. If 1, all tokens are considered. Defaults to None, which means we fall back to the inference server's default value.",
        ),
    ] = None

    top_k: Annotated[
        int | None,
        Field(
            description="Number of top tokens to consider. If -1, all tokens are considered. Defaults to None, which means we fall back to the inference server's default value.",
        ),
    ] = None


class AdvantageConfig(BaseConfig):
    length_weighted_mean: bool = False
    length_weighted_adv: bool = False


class EnvConfig(BaseConfig):
    """Configures an environment for training."""

    id: Annotated[str, Field(description="ID of the environment to use.")] = (
        "reverse-text"
    )
    args: Annotated[
        dict, Field(description="Arguments to pass to the environment.")
    ] = {}
    name: Annotated[
        str | None, Field(description="Name of the environment to use in logs.")
    ] = None


class OrchestratorConfig(BaseSettings):
    """Conf for orchestrator"""

    client: ClientConfig = ClientConfig()
    model: ModelConfig = ModelConfig()
    sampling: SamplingConfig = SamplingConfig()
    env: list[EnvConfig] = [EnvConfig()]
    rollout_transport: Annotated[TransportConfigType, Field(discriminator="type")] = (
        FileSystemTransportConfig()
    )
    batch_size: Annotated[
        int, Field(ge=1, description="Number of samples to train on per step.")
    ] = 128
    rollouts_per_example: Annotated[
        int,
        Field(
            ge=1,
            description="Number of output sequences to return per example during training.",
        ),
    ] = 8
    seq_len: Annotated[int, Field(ge=1)] = 2048
    max_steps: Annotated[
        int | None,
        Field(
            description="Maximum number of training steps to run. If None, will run indefinitely.",
        ),
    ] = None
    max_concurrent: Annotated[
        int | None,
        Field(
            description="Maximum number of concurrent rollouts to generate and score. Will create a global semaphore and pass to verifiers Environment. If None, will not limit concurrency.",
        ),
    ] = None
    max_async_level: Annotated[
        int,
        Field(
            ge=0,
            description="Maximum number of steps the inference can be ahead of training. If 0, will degenerate to synchronous on-policy RL. If >=1, training and inference will be overlapped.",
        ),
    ] = 1
    max_off_policy_steps: Annotated[
        int,
        Field(
            ge=0,
            description="Maximum number of policies that are allowed to generate a single rollout. Rollouts that are generated from more than `max_off_policy_steps` steps ahead of training will be discarded. Higher values yield better throughput, but lead to more off-policyness in training.",
        ),
    ] = 8
    output_dir: Annotated[
        Path,
        Field(
            description="Directory to write outputs to. Will be populated with checkpoints, weights, rollouts and logs as subdirectories. Should be set to a persistent directory with enough disk space. This value should be distinct across experiments running on a single node. See the README for more details."
        ),
    ] = Path("outputs/run_r")

    seed: Annotated[
        int | None, Field(description="Random seed for the orchestrator.")
    ] = 42

    # The logging configuration
    log: LogConfig = LogConfig()

    # The advantage configuration
    advantage: AdvantageConfig | None = AdvantageConfig()

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size % self.rollouts_per_example != 0:
            raise ValueError("batch_size must be divisible by rollouts_per_example")
        return self
