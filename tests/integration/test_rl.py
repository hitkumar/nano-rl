"""Integration test for unified RL launcher"""

import subprocess
from pathlib import Path

import pytest


TIMEOUT = 600  # 10 minutes


@pytest.mark.gpu
@pytest.mark.slow
def test_rl_unified_launcher(tmp_path):
    """Test that the unified RL launcher completes without errors."""
    output_dir = tmp_path / "outputs"
    cmd = [
        "uv", "run", "rl",
        "@", "configs/test/rl.toml",
        "--output-dir", str(output_dir),
    ]
    result = subprocess.run(cmd, timeout=TIMEOUT, capture_output=True, text=True)

    # Check process succeeded
    assert result.returncode == 0, f"RL launcher failed:\n{result.stdout}\n{result.stderr}"

    # Check trainer produced logs
    trainer_log = output_dir / "logs" / "trainer.log"
    assert trainer_log.exists(), "Trainer log not created"
    trainer_output = trainer_log.read_text()
    assert "Step 0" in trainer_output
    assert "Step 2" in trainer_output  # max_steps=3, so steps 0,1,2

    # Check orchestrator produced rollouts
    orchestrator_log = output_dir / "logs" / "orchestrator.log"
    assert orchestrator_log.exists(), "Orchestrator log not created"
    assert "avg_reward" in orchestrator_log.read_text()


@pytest.mark.gpu
@pytest.mark.slow
def test_launcher_detects_trainer_failure(tmp_path):
    """Test that the launcher exits with non-zero when the trainer crashes."""
    output_dir = tmp_path / "outputs"
    cmd = [
        "uv", "run", "rl",
        "@", "configs/test/rl.toml",
        "--output-dir", str(output_dir),
        "--model.name", "nonexistent/model-does-not-exist",
    ]
    result = subprocess.run(cmd, timeout=TIMEOUT, capture_output=True, text=True)
    assert result.returncode != 0


@pytest.mark.gpu
@pytest.mark.slow
def test_launcher_detects_gpu_overlap():
    """Test that the launcher rejects overlapping inference and trainer GPU IDs."""
    cmd = [
        "uv", "run", "rl",
        "@", "configs/test/rl.toml",
        "--inference-gpu-ids", "[0]",
        "--trainer-gpu-ids", "[0]",
    ]
    result = subprocess.run(cmd, timeout=60, capture_output=True, text=True)
    assert result.returncode != 0
    assert "overlap" in result.stderr.lower()


@pytest.mark.gpu
@pytest.mark.slow
def test_rl_multiturn(tmp_path):
    """Test that multi-turn RL training completes without errors."""
    output_dir = tmp_path / "outputs"
    cmd = [
        "uv", "run", "rl",
        "@", "configs/test/rl_multiturn.toml",
        "--output-dir", str(output_dir),
    ]
    result = subprocess.run(cmd, timeout=TIMEOUT, capture_output=True, text=True)

    assert result.returncode == 0, f"Multi-turn RL failed:\n{result.stdout}\n{result.stderr}"

    trainer_log = output_dir / "logs" / "trainer.log"
    assert trainer_log.exists(), "Trainer log not created"
    trainer_output = trainer_log.read_text()
    assert "Step 0" in trainer_output
    assert "Step 2" in trainer_output

    orchestrator_log = output_dir / "logs" / "orchestrator.log"
    assert orchestrator_log.exists(), "Orchestrator log not created"
    assert "avg_reward" in orchestrator_log.read_text()
