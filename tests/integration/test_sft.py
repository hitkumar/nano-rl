"""Integration test for SFT training"""

import json
import subprocess

import pytest


TIMEOUT = 300  # 5 minutes


@pytest.mark.gpu
@pytest.mark.slow
def test_sft_training(tmp_path):
    """Test that SFT training completes and produces expected outputs."""
    output_dir = tmp_path / "outputs"
    cmd = [
        "uv", "run", "torchrun",
        "--nproc-per-node=1",
        "-m", "nano_rl.trainer.sft.train",
        "@", "configs/test/sft.toml",
        "--output-dir", str(output_dir),
    ]
    result = subprocess.run(cmd, timeout=TIMEOUT, capture_output=True, text=True)
    assert result.returncode == 0, f"SFT training failed:\n{result.stdout}\n{result.stderr}"

    # Check metrics were saved
    metrics_file = output_dir / "metrics.json"
    assert metrics_file.exists(), "metrics.json not created"
    metrics = json.loads(metrics_file.read_text())
    assert len(metrics) == 3  # max_steps=3
    assert all("loss/mean" in m for m in metrics)

    # Check final weights were saved
    weights_dir = output_dir / "weights" / "step_3"
    assert weights_dir.exists(), "Final weights not saved"
