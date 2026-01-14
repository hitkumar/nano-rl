import subprocess
import time

import pytest
import verifiers as vf
from nano_rl.orchestrator.utils import set_semaphore
from nano_rl.utils.client import setup_client
from nano_rl.utils.vf import generate_group, generate_rollout
from openai import AsyncOpenAI

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

INFERENCE_STARTUP_TIMEOUT = 300  # 5 minutes to load model


@pytest.fixture(scope="module")
def inference_server():
    """Starts inference server once for the test module, not once for every test"""
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

    yield process  # now tests are run

    # cleanup happens after all the tests are completed
    process.terminate()
    process.wait(timeout=10)


@pytest.fixture(scope="module")
def client(inference_server) -> AsyncOpenAI:
    """
    Create client connected to inference server.
    Inference server is passed to this as this indicates that client depends on inference_server, we attempt to initiate a connection when inference server is started.
    """
    from nano_rl.utils.config import ClientConfig

    config = ClientConfig(timeout=60)
    return setup_client(config)


@pytest.fixture(scope="module")
def env() -> vf.Environment:
    """creates a simple test environment"""
    return vf.ReasoningGymEnv(gym="complex_arithmetic", num_train_examples=10)


async def test_generate_group(client, env):
    """Test that generate group produces valid states"""
    await set_semaphore(10)
    example = env.get_dataset()[0]

    states = await generate_group(
        client=client,
        env=env,
        model_name="Qwen/Qwen3-0.6B",
        example=example,
        rollouts_per_example=2,
        sampling_args={"temperature": 0.2, "max_tokens": 256},
    )

    assert len(states) == 2
    for state in states:
        assert "trajectory" in state
        assert "reward" in state
        assert len(state["trajectory"]) == 1


async def test_generate_rollout(client, env):
    await set_semaphore(1)
    example = env.get_dataset()[0]
    state = await generate_rollout(
        client=client,
        env=env,
        model_name="Qwen/Qwen3-0.6B",
        example=example,
        sampling_args={"temperature": 0.2, "max_tokens": 256},
    )
    tokens = state["trajectory"][0]["tokens"]
    assert "token_ids" in tokens
    assert "trajectory" in state
    assert "reward" in state
    assert len(state["trajectory"]) == 1
