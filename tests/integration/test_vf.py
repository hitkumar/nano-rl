import subprocess
import time

import pytest
import verifiers as vf
from nano_rl.utils.client import setup_clients
from nano_rl.utils.vf import generate_group, generate_rollout

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

INFERENCE_STARTUP_TIMEOUT = 60  # 1 minute to load model


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
def client(inference_server) -> vf.ClientConfig:
    """
    Create client connected to inference server.
    Inference server is passed to this as this indicates that client depends on inference_server, we attempt to initiate a connection when inference server is started.
    """
    from nano_rl.utils.config import ClientConfig

    config = ClientConfig(timeout=60)
    return setup_clients(config)[0]


@pytest.fixture(scope="module")
def env() -> vf.Environment:
    """creates a simple test environment"""
    return vf.load_environment("reverse-text")


async def test_generate_group(client, env):
    """Test that generate group produces valid states"""
    example = env.get_dataset()[0]

    outputs = await generate_group(
        client=client,
        env=env,
        model_name="Qwen/Qwen3-0.6B",
        example=example,
        rollouts_per_example=2,
        sampling_args={"temperature": 0.2, "max_tokens": 256},
    )

    assert len(outputs) == 2
    for output in outputs:
        assert "trajectory" in output
        assert "reward" in output
        assert len(output["trajectory"]) == 1


async def test_generate_rollout(client, env):
    example = env.get_dataset()[0]
    output = await generate_rollout(
        client=client,
        env=env,
        model_name="Qwen/Qwen3-0.6B",
        example=example,
        sampling_args={"temperature": 0.2, "max_tokens": 256},
    )
    assert "trajectory" in output
    assert "reward" in output
    assert len(output["trajectory"]) == 1
