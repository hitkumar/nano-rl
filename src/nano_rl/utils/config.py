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
