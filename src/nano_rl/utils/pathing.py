import asyncio
import time
from pathlib import Path


def get_ckpt_dir(output_dir: Path) -> Path:
    return output_dir / "checkpoints"


def get_weights_dir(output_dir: Path) -> Path:
    return output_dir / "weights"


def get_logs_dir(output_dir: Path) -> Path:
    return output_dir / "logs"


def get_evals_dir(output_dir: Path) -> Path:
    return output_dir / "evals"


def get_rollout_dir(output_dir: Path) -> Path:
    return output_dir / "rollouts"


def get_broadcasts_dir(output_dir: Path) -> Path:
    return output_dir / "broadcasts"


def get_step_path(path: Path, step: int) -> Path:
    return path / f"step_{step}"


def resolve_latest_ckpt_dir(ckpt_dir: Path) -> int | None:
    if not ckpt_dir.exists():
        return None
    step_dirs = list(ckpt_dir.glob("step_*"))
    if step_dirs is None or len(step_dirs) == 0:
        return None
    steps = sorted([int(d.name.split("_")[1]) for d in step_dirs], reverse=True)
    for step in steps:
        if (ckpt_dir / f"step_{step}" / "STABLE").exists():
            return step
    return None


def sync_wait_for_path(path: Path, interval: float = 1.0) -> None:
    while not path.exists():
        time.sleep(interval)


async def wait_for_path(path: Path, interval: float = 1.0) -> None:
    while not path.exists():
        await asyncio.sleep(interval)
