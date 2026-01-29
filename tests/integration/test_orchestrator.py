"""
Integration test for orchestrator.
Tests the full flow: start inference server → generate rollouts → write batches
"""

import subprocess
import time
from pathlib import Path

import pytest
from nano_rl.orchestrator.config import (
    AdvantageConfig,
    EnvConfig,
    OrchestratorConfig,
    SamplingConfig,
)
from nano_rl.orchestrator.orchestrator import orchestrate, state_to_training_sample
from nano_rl.orchestrator.utils import set_semaphore
from nano_rl.transport import setup_training_batch_receiver
from nano_rl.utils.config import ClientConfig

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

INFERENCE_STARTUP_TIMEOUT = 60  # 1 minute to load model


@pytest.fixture(scope="module")
def inference_server():
    """Starts inference server once for the test module"""
    process = subprocess.Popen(
        [
            "uv",
            "run",
            "inference",
            "@",
            "configs/debug/infer.toml",
        ]
    )

    time.sleep(INFERENCE_STARTUP_TIMEOUT)

    yield process

    process.terminate()
    process.wait(timeout=10)


@pytest.fixture
def config(tmp_path: Path, inference_server) -> OrchestratorConfig:
    """Create orchestrator config pointing to test inference server"""
    return OrchestratorConfig(
        output_dir=tmp_path,
        batch_size=8,
        rollouts_per_example=4,
        max_steps=2,
        max_async_level=0,  # synchronous - no weight updates during test
        seq_len=1024,
        sampling=SamplingConfig(temperature=1.0, max_tokens=64),
        advantage=AdvantageConfig(),
        env=[
            EnvConfig(
                id="reverse-text",
            )
        ],
        client=ClientConfig(base_url=["http://localhost:8000/v1"]),
        max_concurrent=4,
    )


async def test_orchestrator_produces_valid_training_samples(
    config: OrchestratorConfig, tmp_path: Path
):
    """Verify that training samples have all required fields"""
    await orchestrate(config)

    receiver = setup_training_batch_receiver(tmp_path, current_step=0)
    batch = receiver.receive()

    # Check ckpt_step is present and valid (should be -1 before any weights are checkpointed)
    assert batch.ckpt_step == -1

    for sample in batch.examples:
        # Check all required fields exist and have correct types
        assert isinstance(sample.prompt_ids, list)
        assert isinstance(sample.prompt_mask, list)
        assert isinstance(sample.completion_ids, list)
        assert isinstance(sample.completion_mask, list)
        assert isinstance(sample.completion_logprobs, list)
        assert isinstance(sample.advantage, float)

        # Check lengths are consistent
        assert len(sample.prompt_ids) == len(sample.prompt_mask)
        assert len(sample.completion_ids) == len(sample.completion_mask)
        assert len(sample.completion_ids) == len(sample.completion_logprobs)

        # Check logprobs are negative (valid log probabilities)
        assert all(lp <= 0 for lp in sample.completion_logprobs)

    advantages = [s.advantage for s in batch.examples]
    rollouts_per_example = config.rollouts_per_example

    # Check each group of rollouts_per_example has mean ≈ 0
    num_groups = len(advantages) // rollouts_per_example
    for i in range(num_groups):
        group = advantages[i * rollouts_per_example : (i + 1) * rollouts_per_example]
        group_mean = sum(group) / len(group)
        assert abs(group_mean) < 0.01, f"Group {i} mean should be ~0, got {group_mean}"
