from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from nano_rl.trainer.config import (
    AdamWConfig,
    CheckpointConfig,
    ConstantSchedulerConfig,
    ModelConfig,
    OptimizerConfigType,
    SchedulerConfigType,
    TokenizerConfig,
)
from nano_rl.utils.config import LogConfig
from nano_rl.utils.pydantic_config import BaseConfig, BaseSettings
from pydantic import BaseModel, Field, model_validator


class BaseDataConfig(BaseModel):
    """Base config for SFT data"""

    batch_size: Annotated[int, Field(ge=1)] = 128
    seq_len: Annotated[int, Field(ge=1)] = 128
    pack_function: Literal["cat", "stack"] = "cat"
    micro_batch_size: Annotated[int, Field(ge=1)] = 1

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("Batch size must be divisible by micro batch size")
        if self.batch_size < self.micro_batch_size:
            raise ValueError(
                "Batch size must be greater than or equal to micro batch size"
            )
        return self


class LossMaskConfig(BaseConfig):
    """Configures which message types contribute to the loss during SFT training"""

    system: Annotated[
        bool, Field(description="Whether system messages contribute to the loss.")
    ] = False
    user: Annotated[
        bool, Field(description="Whether user messages contribute to the loss.")
    ] = False
    assistant: Annotated[
        bool, Field(description="Whether assistant messages contribute to the loss.")
    ] = True
    tool: Annotated[
        bool, Field(description="Whether tool messages contribute to the loss.")
    ] = False


class FakeDataConfig(BaseDataConfig):
    """Configures fake data used for debugging"""

    type: Literal["fake"] = "fake"

    length: Literal["fixed", "variable"] = "fixed"
    input_ids: Literal["increasing", "random"] = "increasing"


class SFTDataConfig(BaseDataConfig):
    """Configures data used for SFT training"""

    type: Literal["sft"] = "sft"
    name: Annotated[
        str, Field(description="Name or path of the HF dataset to use.")
    ] = "PrimeIntellect/Reverse-Text-SFT"
    subsets: Annotated[
        list[str] | None, Field(description="Subsets to use from the HF dataset.")
    ] = None
    splits: Annotated[
        list[str] | None, Field(description="Splits to use from the HF dataset.")
    ] = None
    stopping_strategy: Annotated[
        Literal["first_exhausted", "all_exhausted"],
        Field(description=""),
    ] = "all_exhausted"
    shuffle: Annotated[
        bool,
        Field(
            description="Whether to shuffle the dataset at the beginning of each epoch."
        ),
    ] = True
    seed: Annotated[
        int,
        Field(
            description="Random seed to use for shuffling the dataset. We also shuffle at the end of each epoch by adding epoch count to the seed."
        ),
    ] = 0
    loss_mask: LossMaskConfig = LossMaskConfig()


DataConfigType: TypeAlias = FakeDataConfig | SFTDataConfig


class SFTTrainerConfig(BaseSettings):
    """Configures SFT trainer"""

    model: ModelConfig = ModelConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()
    data: Annotated[DataConfigType, Field(discriminator="type")] = SFTDataConfig()
    optim: Annotated[OptimizerConfigType, Field(discriminator="type")] = AdamWConfig()
    scheduler: Annotated[SchedulerConfigType, Field(discriminator="type")] = (
        ConstantSchedulerConfig()
    )

    output_dir: Annotated[
        Path,
        Field(
            description="Directory to write outputs to. Will be populated with checkpoints and logs as subdirectories. Should be set to a persistent directory with enough disk space. This value should be distinct across experiments running on a single node."
        ),
    ] = Path("outputs")

    max_steps: Annotated[
        int | None,
        Field(
            description="Maximum number of steps to run training for. If None, will run indefinitely."
        ),
    ] = None

    loss_impl: Annotated[
        Literal["liger", "torch"],
        Field(description="Implementation of the cross entropy loss function to use."),
    ] = "torch"

    dist_timeout_seconds: Annotated[
        int,
        Field(
            description="Timeout in seconds for torch distributed ops. Defaults to 600 seconds.",
        ),
    ] = 600

    log: LogConfig = LogConfig()

    output_dir: Annotated[
        Path,
        Field(
            description="Directory to write outputs to. Will be populated with checkpoints and logs as subdirectories. Should be set to a persistent directory with enough disk space. This value should be distinct across experiments running on a single node. See the README for more details."
        ),
    ] = Path("outputs")

    ckpt: CheckpointConfig | None = CheckpointConfig()

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        if self.tokenizer.name is None:
            self.tokenizer.name = self.model.name
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.model.trust_remote_code
        return self
