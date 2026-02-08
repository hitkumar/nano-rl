"""Unit tests for RLConfig validators and propagation"""

import pytest
from nano_rl.inference.config import InferenceConfig
from nano_rl.rl_config import RLConfig, SharedModelConfig, SharedWeightBroadcastConfig


def _make_config(**kwargs):
    """Create RLConfig with inference enabled by default."""
    defaults = dict(
        inference_gpu_ids=[0],
        trainer_gpu_ids=[1],
    )
    defaults.update(kwargs)
    return RLConfig(**defaults)


def test_model_name_propagates():
    config = _make_config(model=SharedModelConfig(name="test/model"))
    assert config.trainer.model.name == "test/model"
    assert config.orchestrator.model.name == "test/model"


def test_max_steps_propagates():
    config = _make_config(max_steps=50)
    assert config.trainer.max_steps == 50
    assert config.orchestrator.max_steps == 50


def test_seq_len_propagates():
    config = _make_config(seq_len=4096)
    assert config.trainer.model.seq_len == 4096
    assert config.orchestrator.seq_len == 4096


def test_async_level_propagates():
    config = _make_config(max_async_level=2)
    assert config.trainer.max_async_level == 2
    assert config.orchestrator.max_async_level == 2


def test_output_dir_propagates():
    config = _make_config(output_dir="outputs/test")
    assert str(config.trainer.output_dir) == "outputs/test"
    assert str(config.orchestrator.output_dir) == "outputs/test"


def test_log_level_propagates():
    config = _make_config()
    assert config.trainer.log.level == "info"
    assert config.orchestrator.log.level == "info"


def test_nccl_broadcast_propagates():
    config = _make_config(weight_broadcast=SharedWeightBroadcastConfig(type="nccl"))
    assert config.trainer.weight_broadcast.type == "nccl"
    assert config.orchestrator.weight_broadcast.type == "nccl"
    assert config.trainer.weight_broadcast.inference_world_size == 1


def test_filesystem_broadcast_propagates():
    config = _make_config(weight_broadcast=SharedWeightBroadcastConfig(type="filesystem"))
    assert config.trainer.weight_broadcast.type == "filesystem"
    assert config.orchestrator.weight_broadcast.type == "filesystem"


def test_inference_dp_auto_configured():
    config = _make_config(
        inference_gpu_ids=[0, 1, 2, 3],
        trainer_gpu_ids=[4],
        inference=InferenceConfig(),
    )
    assert config.inference.parallel.dp == 4


def test_orchestrator_client_urls_match_inference():
    config = _make_config(
        inference_gpu_ids=[0, 1],
        trainer_gpu_ids=[2],
        inference=InferenceConfig(),
    )
    urls = config.orchestrator.client.base_url
    assert len(urls) == 2
    assert "8000" in urls[0]
    assert "8001" in urls[1]


def test_gpu_overlap_raises():
    with pytest.raises(Exception, match="shared between trainer and inference"):
        _make_config(inference_gpu_ids=[0, 1], trainer_gpu_ids=[1, 2])


def test_nccl_single_gpu_raises():
    with pytest.raises(Exception, match="at least 2 GPUs"):
        RLConfig(
            inference_gpu_ids=[],
            trainer_gpu_ids=[0],
            inference=None,
            weight_broadcast=SharedWeightBroadcastConfig(type="nccl"),
        )


def test_seq_len_trainer_less_than_orch_raises():
    """Trainer seq_len must be >= orchestrator seq_len."""
    with pytest.raises(Exception, match="seq_len"):
        RLConfig(
            inference_gpu_ids=[0],
            trainer_gpu_ids=[1],
            # Don't use shared seq_len — set them independently to create mismatch
            trainer={"model": {"seq_len": 512}},
            orchestrator={"seq_len": 2048},
        )
