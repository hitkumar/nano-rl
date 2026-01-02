import pytest
from nano_rl.trainer.parallel_dims import ParallelDims


def test_validation_auto_dp_shard():
    """dp_shard=-1 auto-computes correctly."""
    pd = ParallelDims(
        dp_replicate=2,
        dp_shard=-1,
        cp=2,
        tp=2,
        pp=1,
        ep=1,
        world_size=32,
    )
    # dp_shard = 32 // (2 * 2 * 2 * 1) = 4
    assert pd.dp_shard == 4


def test_properties():
    """Property flags work correctly."""
    pd = ParallelDims(
        dp_replicate=2,
        dp_shard=4,
        cp=2,
        tp=1,
        pp=1,
        ep=4,
        world_size=16,
    )
    assert pd.dp_enabled
    assert pd.dp_replicate_enabled
    assert pd.dp_shard_enabled
    assert pd.cp_enabled
    assert not pd.tp_enabled
    assert not pd.pp_enabled
    assert pd.ep_enabled
    assert pd.fsdp_enabled
    assert pd.dp_degree == 8
    assert pd.non_data_parallel_size == 2  # cp * tp * pp
