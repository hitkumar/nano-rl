"""
Unified RL launcher that does inference, training, and orchestrator.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import tomli_w
from nano_rl.rl_config import RLConfig
from nano_rl.utils.logger import get_logger, setup_logger
from nano_rl.utils.pathing import (
    get_logs_dir,
    get_rollout_dir,
    get_temp_toml_file,
    get_weights_dir,
)
from nano_rl.utils.pydantic_config import parse_argv


def clean_directories(config: RLConfig) -> None:
    """Removes rollouts, weights and logs directories if clean is True"""
    dirs_to_clean = [
        get_rollout_dir(config.output_dir),
        get_weights_dir(config.output_dir),
        get_logs_dir(config.output_dir),
    ]
    for d in dirs_to_clean:
        if d.exists():
            shutil.rmtree(d)


def write_component_config(config: dict) -> Path:
    temp_toml_path = get_temp_toml_file()
    with open(temp_toml_path, "wb") as f:
        tomli_w.dump(config, f)
    return temp_toml_path


def build_env_with_gpus(gpu_ids: list[int]) -> dict:
    """Build environment variables for trainer/inference process."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    return env


def start_inference_server(config: RLConfig, log_dir: Path) -> subprocess.Popen | None:
    """Start the inference server subprocess."""
    if config.inference is None:
        return None

    inference_dict = config.inference.model_dump(exclude_none=True, mode="json")
    config_path = write_component_config(inference_dict)

    cmd = ["uv", "run", "inference", "@", str(config_path)]
    env = build_env_with_gpus(config.inference_gpu_ids)
    log_file = open(log_dir / "inference.log", "w")
    return subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)


def start_orchestrator(config: RLConfig, log_dir: Path) -> subprocess.Popen:
    """Start the orchestrator subprocess."""
    assert config.orchestrator is not None
    orch_dict = config.orchestrator.model_dump(exclude_none=True, mode="json")
    config_path = write_component_config(orch_dict)
    cmd = ["uv", "run", "orchestrator", "@", str(config_path)]

    log_file = open(log_dir / "orchestrator.log", "w")
    return subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def start_trainer(config: RLConfig, log_dir: Path) -> subprocess.Popen:
    """Start the trainer subprocess."""
    trainer_dict = config.trainer.model_dump(exclude_none=True, mode="json")
    config_path = write_component_config(trainer_dict)

    nproc = len(config.trainer_gpu_ids)
    cmd = [
        "uv",
        "run",
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--master_port=29500",
        "-m",
        "nano_rl.trainer.rl.train",
        "@",
        str(config_path),
    ]
    env = build_env_with_gpus(config.trainer_gpu_ids)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log_file = open(log_dir / "trainer.log", "w")
    return subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)


def wait_for_inference_ready(config: RLConfig, timeout: float = 120) -> None:
    """Wait for inference server endpoint to be ready.

    Note: vLLM's run_multi_api_server uses SO_REUSEPORT so all API server
    processes share the same port. We only need to check the single port.
    """
    import urllib.error
    import urllib.request

    if config.inference is None:
        return

    port = config.inference.server.port or 8000
    host = config.inference.server.host or "localhost"
    url = f"http://{host}:{port}/health"

    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=1)
            print(f"Inference server {url} is ready")
            return
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(1)

    raise TimeoutError(f"Timeout waiting for inference server {url} to be ready")


def cleanup_processes(processes: list[subprocess.Popen]) -> None:
    """Terminate all subprocesses gracefully."""
    for p in processes:
        if p.poll() is None:  # Still running
            p.terminate()
    for p in processes:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def main() -> None:
    config = parse_argv(RLConfig)
    if config.clean:
        clean_directories(config)

    log_dir = get_logs_dir(config.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(
        config.log.level,
        log_file=(log_dir / "rl.log"),
    )
    logger = get_logger()
    processes: list[subprocess.Popen] = []

    try:
        logger.info("Starting inference server")
        inference_proc = start_inference_server(config, log_dir)
        if inference_proc:
            processes.append(inference_proc)
            wait_for_inference_ready(config)
        else:
            logger.info("No inference config, assuming external server")

        logger.info("Starting orchestrator")
        orch_process = start_orchestrator(config, log_dir)
        processes.append(orch_process)

        train_process = start_trainer(config, log_dir)
        processes.append(train_process)

        logger.info("All processes started, waiting for trainer to finish")

        # waiting for trainer to finish as that is the main workload
        train_process.wait()

        logger.info("Training done")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, terminating processes")
    except Exception as e:
        logger.exception("Exception occurred, terminating processes")
    finally:
        cleanup_processes(processes)


if __name__ == "__main__":
    main()
