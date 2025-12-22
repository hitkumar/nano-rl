from typing import Annotated, Literal, TypeAlias

from nano_rl.utils.pydantic_config import BaseConfig
from pydantic import BaseModel, Field, model_validator

AttnImplementation: TypeAlias = Literal[
    "sdpa", "flash_attention_2", "flash_attention_3"
]


class TokenizerConfig(BaseConfig):
    """Configuration for the tokenizer."""

    name: Annotated[
        str | None,
        Field(
            description="The name or path of the tokenizer to use. If None, will use the model's default tokenizer."
        ),
    ] = None

    trust_remote_code: Annotated[
        bool | None,
        Field(
            description="Whether to trust remote code for tokenizer initialization. If None, will use the model's default trust remote code setting.",
        ),
    ] = None

    chat_template: Annotated[
        str | None,
        Field(
            description="The chat template to use for the tokenizer. If None, will use the tokenizer's default chat template."
        ),
    ] = None


class BaseOptimizerConfig(BaseConfig):
    lr: Annotated[float, Field(ge=0)] = 1e-6
    weight_decay: Annotated[float, Field(ge=0)] = 0.01
    max_norm: Annotated[
        float, Field(ge=0, description="Maximum gradient norm to clip")
    ] = 1.0


class AdamWConfig(BaseModel):
    type: Literal["adamw"] = "adamw"
    betas1: Annotated[float, Field(ge=0)] = 0.9
    betas2: Annotated[float, Field(ge=0)] = 0.999


class MuonConfig(BaseModel):
    type: Literal["muon"] = "muon"
    betas1: Annotated[float, Field(ge=0)] = 0.9
    betas2: Annotated[float, Field(ge=0)] = 0.999


class SGDConfig(BaseModel):
    type: Literal["sgd"] = "sgd"
    nesterov: bool = True
    momentum: float = 0.9


OptimizerConfigType: TypeAlias = AdamWConfig | SGDConfig | MuonConfig


class ConstantSchedulerConfig(BaseModel):
    """Configuration for constant learning rate scheduler."""

    type: Literal["constant"] = "constant"


SchedulerConfigType: TypeAlias = ConstantSchedulerConfig


class ActivationCheckpointConfig(BaseConfig):
    """Configures activation checkpointing."""

    freq: Annotated[
        int,
        Field(
            ge=1,
            description="Applies activation checkpointing to every `freq` layers. Defaults to 1, which will is full activation checkpointing.",
        ),
    ] = 1


class CompileConfig(BaseConfig):
    """Configures model compilation."""

    fullgraph: Annotated[
        bool,
        Field(description="Whether to compile the transformer blocks with fullgraph."),
    ] = False


class ModelConfig(BaseConfig):
    """Model definition"""

    name: Annotated[str, Field(description="Name or path of the hf model to use")] = (
        "Qwen/Qwen3-0.6B"
    )

    seq_len: Annotated[
        int, Field(description="The sequence length to use for the model.")
    ] = 2048

    attn: Annotated[
        AttnImplementation, Field(description="The attention implementation to use.")
    ] = "flash_attention_2"

    compile: Annotated[
        CompileConfig | None,
        Field(
            description="Compile settings to use while compiling the model using torch.compile"
        ),
    ] = None

    ac: Annotated[
        ActivationCheckpointConfig | None,
        Field(
            description="Whether to apply activation checkpointing to the model. If None, will not apply activation checkpointing.",
        ),
    ] = None

    trust_remote_code: Annotated[
        bool,
        Field(
            description="Whether to trust remote code for model and tokenizer initialization.",
        ),
    ] = False

    impl: Annotated[
        Literal["hf", "liger_kernel", "custom"],
        Field(description="The implementation to use for the model."),
    ] = "hf"

    @model_validator(mode="after")
    def trust_remote_code_only_hf(self):
        """
        Trust remote code only if it is from HF
        """
        if self.trust_remote_code:
            if self.impl != "hf":
                raise ValueError(
                    f"Trust remove code only for HF impl, this one is {self.impl}"
                )

        return self
