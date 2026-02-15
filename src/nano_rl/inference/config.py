from argparse import Namespace
from typing import Annotated, Literal

from nano_rl.utils.pydantic_config import BaseConfig, BaseSettings
from pydantic import Field, model_validator


class ServerConfig(BaseConfig):
    host: Annotated[str | None, Field(description="The host to run the server on.")] = (
        "localhost"
    )
    port: Annotated[int | None, Field(description="The port to run the server on.")] = (
        8000
    )


class ParallelConfig(BaseConfig):
    tp: Annotated[int, Field(ge=1, description="Tensor parallel size.")] = 1
    dp: Annotated[int, Field(ge=1, description="Data parallel size.")] = 1

    def __str__(self) -> str:
        return f"tp={self.tp} dp={self.dp}"


class ModelConfig(BaseConfig):
    name: Annotated[
        str,
        Field(
            description="Name or path of the HF model to use.",
        ),
    ] = "Qwen/Qwen3-0.6B"
    dtype: Annotated[
        Literal["auto", "float16", "bfloat16", "float32"],
        Field(
            description="Data type for model weights and activations. If 'auto' will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models. Passed to vLLM as `--dtype`",
        ),
    ] = "auto"
    max_model_len: Annotated[
        int | None,
        Field(
            description="Maximum model context length. If None, will use the maximum context length from model config. Passed to vLLM as `--max-model-len`",
        ),
    ] = None
    trust_remote_code: Annotated[
        bool,
        Field(
            description="Whether to trust remote code. Passed to vLLM engine init",
        ),
    ] = False
    enforce_eager: Annotated[
        bool,
        Field(
            description="Whether to enforce eager mode. If False, will use PyTorch eager and cuda graphs in hybrid for maximal performance. Passed to vLLM as `--enforce-eager`",
        ),
    ] = False


class InferenceConfig(BaseSettings):
    """Configures inference settings. We use BaseSettings here to allow cmd line and env variable overrides."""

    server: ServerConfig = ServerConfig()
    model: ModelConfig = ModelConfig()
    parallel: ParallelConfig = ParallelConfig()
    gpu_memory_utilization: Annotated[float, Field(ge=0.0, le=1.0)] = 0.9
    api_server_count: Annotated[
        int,
        Field(
            ge=1,
            description="The number of API servers to spawn. Passed to vLLM as `--api-server-count`. "
            "When using DP, this should match dp to have one API server per DP replica.",
        ),
    ] = 1
    seed: Annotated[
        int,
        Field(
            description="Seed the inference components. Passed to vLLM as `--seed`",
        ),
    ] = 0

    @model_validator(mode="after")
    def auto_setup_api_server_count(self):
        """Ensure api_server_count matches dp for multi-server setups."""
        if self.api_server_count < self.parallel.dp:
            self.api_server_count = self.parallel.dp
        return self

    def to_vllm_args(self) -> Namespace:
        return Namespace(
            host=self.server.host,
            port=self.server.port,
            model=self.model.name,
            dtype=self.model.dtype,
            max_model_len=self.model.max_model_len,
            trust_remote_code=self.model.trust_remote_code,
            enforce_eager=self.model.enforce_eager,
            tensor_parallel_size=self.parallel.tp,
            data_parallel_size=self.parallel.dp,
            gpu_memory_utilization=self.gpu_memory_utilization,
            api_server_count=self.api_server_count,
            seed=self.seed,
            # we don't want the raw logits, but actual probabilities
            logprobs_mode="processed_logprobs",
        )
