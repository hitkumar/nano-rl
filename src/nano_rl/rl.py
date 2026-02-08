"""
Unified RL launcher that does inference, training, and orchestrator.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread

import tomli_w
import pynvml
from nano_rl.rl_config import RLConfig
from nano_rl.utils.logger import get_logger, setup_logger
from nano_rl.utils.pathing import (
    get_broadcasts_dir,
    get_logs_dir,
    get_rollout_dir,
    get_temp_toml_file,
    get_weights_dir,
)
from nano_rl.utils.pydantic_config import parse_argv


def check_gpus_available(gpu_ids: list[int]) -> None:
    """Raise error if there are existing processes on the specified GPUs."""
    pynvml.nvmlInit()
    occupied = []
    for gpu_id in gpu_ids:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if processes:
            pids = [p.pid for p in processes]
            occupied.append((gpu_id, pids))
    if occupied:
        msg = "Existing processes found on GPUs:\n"
        for gpu_id, pids in occupied:
            msg += f"  GPU {gpu_id}: PIDs {pids}\n"
        msg += "Kill these processes or use different GPUs."
        raise RuntimeError(msg)


def clean_directories(config: RLConfig) -> None:
    """Removes rollouts, weights, broadcasts and logs directories if clean is True"""
    dirs_to_clean = [
        get_rollout_dir(config.output_dir),
        get_weights_dir(config.output_dir),
        get_broadcasts_dir(config.output_dir),
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


def start_inference_servers(config: RLConfig, log_dir: Path) -> list[subprocess.Popen]:
    """Start separate inference server subprocesses, one per DP replica.

    Each server runs on a different port and GPU set, allowing individual
    addressing for weight updates (unlike SO_REUSEPORT which shares a port).
    """
    if config.inference is None:
        return []

    processes = []
    base_port = config.inference.server.port or 8000
    tp = config.inference.parallel.tp
    dp = config.inference.parallel.dp

    for dp_rank in range(dp):
        # Calculate GPU IDs for this DP replica
        start_idx = dp_rank * tp
        end_idx = start_idx + tp
        gpu_ids = config.inference_gpu_ids[start_idx:end_idx]

        # Create config for this server instance
        inference_dict = config.inference.model_dump(exclude_none=True, mode="json")
        inference_dict["server"]["port"] = base_port + dp_rank
        inference_dict["parallel"]["dp"] = 1  # Each server is a single DP replica
        inference_dict["api_server_count"] = 1  # Single API server per process

        config_path = write_component_config(inference_dict)
        cmd = ["uv", "run", "inference", "@", str(config_path)]
        env = build_env_with_gpus(gpu_ids)

        log_file = open(log_dir / f"inference_{dp_rank}.log", "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append(proc)

    return processes


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


def start_trainer(config: RLConfig, log_dir: Path) -> tuple[subprocess.Popen, Path]:
    """Start the trainer subprocess. Returns (process, log_file_path)."""
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
    trainer_log_path = log_dir / "trainer.log"
    log_file = open(trainer_log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, trainer_log_path


def wait_for_inference_ready(config: RLConfig, timeout: float = 120) -> None:
    """Wait for all inference server endpoints to be ready."""
    import urllib.error
    import urllib.request

    if config.inference is None:
        return

    base_port = config.inference.server.port or 8000
    host = config.inference.server.host or "localhost"
    dp = config.inference.parallel.dp

    start = time.time()
    for dp_rank in range(dp):
        port = base_port + dp_rank
        url = f"http://{host}:{port}/health"

        while time.time() - start < timeout:
            try:
                urllib.request.urlopen(url, timeout=1)
                print(f"Inference server {url} is ready")
                break
            except (urllib.error.URLError, ConnectionRefusedError):
                time.sleep(1)
        else:
            raise TimeoutError(
                f"Timeout waiting for inference server {url} to be ready"
            )


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


def cleanup_threads(threads: list[Thread]) -> None:
    """Wait for all monitor threads to finish."""
    for thread in threads:
        thread.join(timeout=5)


def monitor_process(
    process: subprocess.Popen,
    stop_event: Event,
    error_queue: list[Exception],
    process_name: str,
) -> None:
    """Monitor a subprocess in a daemon thread.

    Blocks on process.wait() until the subprocess exits.
    If non-zero exit code, appends an error to the shared error_queue.
    Always sets stop_event so the main loop knows this process is done.
    """
    process.wait()
    if process.returncode != 0:
        error_queue.append(
            RuntimeError(f"{process_name} failed with exit code {process.returncode}")
        )
    stop_event.set()


def start_monitor_thread(
    process: subprocess.Popen,
    name: str,
    stop_events: dict[str, Event],
    error_queue: list[Exception],
    monitor_threads: list[Thread],
) -> None:
    """Spawn a daemon thread that watches a subprocess for exit."""
    stop_event = Event()
    stop_events[name] = stop_event
    thread = Thread(
        target=monitor_process,
        args=(process, stop_event, error_queue, name),
        daemon=True,
    )
    thread.start()
    monitor_threads.append(thread)


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

    all_gpu_ids = list(set(config.inference_gpu_ids + config.trainer_gpu_ids))
    check_gpus_available(all_gpu_ids)

    overlap = set(config.inference_gpu_ids) & set(config.trainer_gpu_ids)
    if overlap:
        raise ValueError(
            f"Inference and trainer GPU IDs overlap: {sorted(overlap)}. "
            f"They must use separate GPUs."
        )

    processes: list[subprocess.Popen] = []
    monitor_threads: list[Thread] = []
    error_queue: list[Exception] = []
    stop_events: dict[str, Event] = {}

    try:
        logger.info("Starting inference servers")
        inference_procs = start_inference_servers(config, log_dir)
        if inference_procs:
            processes.extend(inference_procs)
            for i, proc in enumerate(inference_procs):
                start_monitor_thread(
                    proc, f"inference_{i}", stop_events, error_queue, monitor_threads
                )
            wait_for_inference_ready(config)
        else:
            logger.info("No inference config, assuming external server")

        logger.info("Starting orchestrator")
        orch_process = start_orchestrator(config, log_dir)
        processes.append(orch_process)
        start_monitor_thread(
            orch_process, "orchestrator", stop_events, error_queue, monitor_threads
        )

        train_process, trainer_log_path = start_trainer(config, log_dir)
        processes.append(train_process)
        start_monitor_thread(
            train_process, "trainer", stop_events, error_queue, monitor_threads
        )

        logger.info("All processes started, showing trainer logs...")

        # Tail trainer logs to terminal so the user sees progress
        tail_process = subprocess.Popen(["tail", "-F", str(trainer_log_path)])
        processes.append(tail_process)

        # Poll for errors until trainer exits.
        # Orchestrator never exits on its own (update_policy_loop keeps it alive),
        # but if it crashes, error_queue catches it.
        while not stop_events["trainer"].is_set():
            if error_queue:
                logger.error(f"Process error: {error_queue[0]}")
                cleanup_threads(monitor_threads)
                cleanup_processes(processes)
                sys.exit(1)
            time.sleep(1)

        # Check if trainer failed with an error
        if error_queue:
            logger.error(f"Process error: {error_queue[0]}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            sys.exit(1)

        logger.info("Training finished")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, terminating processes")
        sys.exit(1)
    except Exception as e:
        logger.exception("Exception occurred, terminating processes")
        sys.exit(1)
    finally:
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)


if __name__ == "__main__":
    main()
