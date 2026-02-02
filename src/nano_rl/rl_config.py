"""
Unified RL launcher configuration.

This module defines the RLConfig that wraps trainer, orchestrator, and inference
configs, automatically propagating shared settings across components.
"""

from pathlib import Path
from typing import Annotated, Literal

from nano_rl.inference.config import InferenceConfig
from nano_rl.orchestrator.config import (
    FileSystemWeightBroadcastConfig as OrchestratorFileSystemWeightBroadcastConfig,
    NCCLWeightBroadcastConfig as OrchestratorNCCLWeightBroadcastConfig,
    OrchestratorConfig,
)
from nano_rl.trainer.rl.config import (
    FileSystemWeightBroadcastConfig as TrainerFileSystemWeightBroadcastConfig,
    NCCLWeightBroadcastConfig as TrainerNCCLWeightBroadcastConfig,
    RlTrainerConfig,
)
from nano_rl.utils.pydantic_config import BaseConfig, BaseSettings
from pydantic import Field, model_validator


class SharedLogConfig(BaseConfig):
    """Configures shared logging across all components."""

    level: Annotated[
        str | None,
        Field(description="The log level to use (debug, info, warning, error)."),
    ] = "info"

    file: Annotated[
        bool | None,
        Field(description="Whether to log to a file."),
    ] = True


class SharedCheckpointConfig(BaseConfig):
    """Configures shared checkpoint settings."""

    interval: Annotated[
        int | None,
        Field(description="The interval at which to save checkpoints."),
    ] = None


class SharedModelConfig(BaseConfig):
    """Configures shared model settings."""

    name: Annotated[
        str,
        Field(description="The name of the model to use."),
    ] = "Qwen/Qwen3-0.6B"


class SharedWeightBroadcastConfig(BaseConfig):
    """Configures shared weight broadcast settings."""

    type: Annotated[
        Literal["nccl", "filesystem"],
        Field(description="The type of weight broadcast to use."),
    ] = "filesystem"


class RLConfig(BaseSettings):
    """
    Unified configuration for an RL training run.

    This config wraps trainer, orchestrator, and inference configs,
    automatically propagating shared settings across components via
    Pydantic validators.
    """

    # =========================================================================
    # Submodule configurations
    # =========================================================================

    trainer: RlTrainerConfig = RlTrainerConfig()

    orchestrator: OrchestratorConfig = OrchestratorConfig()

    inference: Annotated[
        InferenceConfig | None,
        Field(
            description="The inference config. If None, will not start an inference process. "
            "Only viable if an inference server was started manually."
        ),
    ] = None

    # =========================================================================
    # Top-level configurations
    # =========================================================================

    log: Annotated[
        SharedLogConfig,
        Field(description="Shared log configs for all components."),
    ] = SharedLogConfig()

    clean: Annotated[
        bool,
        Field(
            description="Whether to clean the rollouts, weights, and logs directories at startup. "
            "If True, will irreversibly delete these directories."
        ),
    ] = True

    inference_gpu_ids: Annotated[
        list[int],
        Field(description="The GPU IDs to use for inference server(s)."),
    ] = [0]

    trainer_gpu_ids: Annotated[
        list[int],
        Field(description="The GPU IDs to use for the trainer."),
    ] = [1]

    # =========================================================================
    # Shared configurations (propagated to submodules)
    # =========================================================================

    output_dir: Annotated[
        Path,
        Field(
            description="The directory to store outputs. "
            "Should typically be set to an experiment identifier."
        ),
    ] = Path("outputs/rl")

    ckpt: Annotated[
        SharedCheckpointConfig | None,
        Field(
            description="Shared checkpoint configs. Propagated to trainer and orchestrator."
        ),
    ] = None

    model: Annotated[
        SharedModelConfig | None,
        Field(description="Shared model configs. Propagated to all components."),
    ] = None

    max_steps: Annotated[
        int | None,
        Field(
            description="Maximum training steps. Propagated to trainer and orchestrator."
        ),
    ] = None

    seq_len: Annotated[
        int | None,
        Field(
            description="Sequence length. If set, configures both trainer.model.seq_len "
            "and orchestrator.seq_len."
        ),
    ] = None

    max_async_level: Annotated[
        int | None,
        Field(description="Async level. Propagated to trainer and orchestrator."),
    ] = None

    weight_broadcast: Annotated[
        SharedWeightBroadcastConfig | None,
        Field(description="Weight broadcast config. Propagated to all components."),
    ] = None

    # =========================================================================
    # Validators - auto-propagate shared settings
    # =========================================================================

    @model_validator(mode="after")
    def auto_setup_inference_dp(self):
        """Auto-configure inference DP based on GPU count. Must run early since other validators depend on it."""
        if (
            self.inference
            and len(self.inference_gpu_ids)
            != self.inference.parallel.dp * self.inference.parallel.tp
        ):
            tp = self.inference.parallel.tp
            if len(self.inference_gpu_ids) % tp != 0:
                raise ValueError(
                    f"Number of inference GPUs ({len(self.inference_gpu_ids)}) "
                    f"must be divisible by tensor parallel size ({tp})"
                )
            self.inference.parallel.dp = len(self.inference_gpu_ids) // tp
            self.inference.api_server_count = self.inference.parallel.dp
        return self

    @model_validator(mode="after")
    def auto_setup_logs(self):
        """Propagate shared log config to trainer and orchestrator."""
        if self.log is not None:
            if self.log.level is not None:
                self.trainer.log.level = self.log.level
                self.orchestrator.log.level = self.log.level
            if self.log.file is not None:
                self.trainer.log.file = self.log.file
                self.orchestrator.log.file = self.log.file
        return self

    @model_validator(mode="after")
    def auto_setup_output_dir(self):
        """Propagate output_dir to trainer and orchestrator."""
        if self.output_dir is not None:
            self.trainer.output_dir = self.output_dir
            self.orchestrator.output_dir = self.output_dir
        return self

    @model_validator(mode="after")
    def auto_setup_model(self):
        """Propagate model name to all components."""
        if self.model is not None:
            self.trainer.model.name = self.model.name
            self.orchestrator.model.name = self.model.name
            if self.inference is not None:
                self.inference.model.name = self.model.name
        return self

    @model_validator(mode="after")
    def auto_setup_max_steps(self):
        """Propagate max_steps to trainer and orchestrator."""
        if self.max_steps is not None:
            self.trainer.max_steps = self.max_steps
            self.orchestrator.max_steps = self.max_steps
        return self

    @model_validator(mode="after")
    def auto_setup_seq_len(self):
        """Propagate seq_len to trainer and orchestrator."""
        if self.seq_len is not None:
            self.trainer.model.seq_len = self.seq_len
            self.orchestrator.seq_len = self.seq_len
        return self

    @model_validator(mode="after")
    def auto_setup_async_level(self):
        """Propagate max_async_level to trainer and orchestrator."""
        if self.max_async_level is not None:
            self.trainer.max_async_level = self.max_async_level
            self.orchestrator.max_async_level = self.max_async_level
        return self

    @model_validator(mode="after")
    def auto_setup_weight_broadcast(self):
        """Configure weight broadcast for all components based on type."""
        broadcast_type = self.weight_broadcast.type if self.weight_broadcast else "filesystem"

        if broadcast_type == "nccl":
            # Calculate inference world size from inference config
            inference_world_size = (
                self.inference.parallel.dp * self.inference.parallel.tp
                if self.inference
                else 1
            )
            self.trainer.weight_broadcast = TrainerNCCLWeightBroadcastConfig(
                inference_world_size=inference_world_size
            )
            self.orchestrator.weight_broadcast = (
                OrchestratorNCCLWeightBroadcastConfig()
            )
        else:  # filesystem
            self.trainer.weight_broadcast = TrainerFileSystemWeightBroadcastConfig()
            self.orchestrator.weight_broadcast = (
                OrchestratorFileSystemWeightBroadcastConfig()
            )
        return self

    @model_validator(mode="after")
    def auto_setup_ckpt(self):
        """Propagate checkpoint config to trainer."""
        if self.ckpt is not None:
            if self.ckpt.interval is not None:
                self.trainer.ckpt.interval = self.ckpt.interval
        return self

    @model_validator(mode="after")
    def auto_setup_orchestrator_client(self):
        """Auto-configure orchestrator client URLs based on inference config.

        Creates one URL per DP replica, each on a different port, allowing
        the orchestrator to address each inference server individually for
        weight updates.
        """
        if self.inference is not None:
            base_port = self.inference.server.port or 8000
            host = self.inference.server.host or "localhost"
            dp = self.inference.parallel.dp
            self.orchestrator.client.base_url = [
                f"http://{host}:{base_port + dp_rank}/v1" for dp_rank in range(dp)
            ]
        return self

    # =========================================================================
    # Validation
    # =========================================================================

    @model_validator(mode="after")
    def validate_gpu_allocation(self):
        """Ensure no GPU ID overlap between trainer and inference."""
        trainer_set = set(self.trainer_gpu_ids)
        inference_set = set(self.inference_gpu_ids)
        overlap = trainer_set & inference_set
        if overlap:
            raise ValueError(
                f"GPU IDs cannot be shared between trainer and inference. "
                f"Overlapping GPUs: {overlap}"
            )
        return self

    @model_validator(mode="after")
    def validate_nccl_requirements(self):
        """Validate NCCL broadcast requirements."""
        if self.weight_broadcast and self.weight_broadcast.type == "nccl":
            # NCCL requires at least 2 GPUs
            num_gpus = len(set(self.trainer_gpu_ids + self.inference_gpu_ids))
            if num_gpus < 2:
                raise ValueError(
                    "NCCL weight broadcast requires at least 2 GPUs "
                    "to build the broadcast process group."
                )
        return self

    @model_validator(mode="after")
    def validate_seq_len(self):
        """Ensure trainer seq_len >= orchestrator seq_len."""
        if self.trainer.model.seq_len < self.orchestrator.seq_len:
            raise ValueError(
                f"Trainer model seq_len ({self.trainer.model.seq_len}) must be >= "
                f"orchestrator seq_len ({self.orchestrator.seq_len}). "
                "The trainer needs to handle sequences at least as long as those from the orchestrator."
            )
        return self
