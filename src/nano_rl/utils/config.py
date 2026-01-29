from typing import Annotated, Literal

from nano_rl.utils.pydantic_config import BaseConfig
from pydantic import Field


class LogConfig(BaseConfig):
    """Configures the logger"""

    level: Annotated[
        str,
        Field(
            description="Logging level for the process. Will determine the logging verbosity and format."
        ),
    ] = "info"

    vf_level: Annotated[
        str,
        Field(
            description="Logging level for the verifiers package. Will determine the logging verbosity and format."
        ),
    ] = "warn"

    file: Annotated[
        bool,
        Field(
            description="Whether to log to a file. If True, will log to a file in the output directory.",
        ),
    ] = True

    log_data: Annotated[
        bool,
        Field(
            description="Whether to log the first data sample to the logger.",
        ),
    ] = False


class ModelConfig(BaseConfig):
    """Configures the model."""

    name: Annotated[str, Field(description="Name or path of the HF model to use.")] = (
        "Qwen/Qwen3-0.6B"
    )

    trust_remote_code: Annotated[
        bool,
        Field(
            description="Whether to trust remote code for tokenizer initialization.",
        ),
    ] = False


class ClientConfig(BaseConfig):
    # Not adding apikey for now as we assume local inference server
    base_url: Annotated[
        list[str],
        Field(
            description="Base URLs for inference servers. If multiple URLs provided, requests will round-robin across servers."
        ),
    ] = ["http://localhost:8000/v1"]
    timeout: Annotated[float, Field(description="Request timeout.")] = 1200.0
