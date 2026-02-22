from unittest.mock import MagicMock

import verifiers as vf

from nano_rl.orchestrator.trajectories import interleave_rollout


def _make_step(prompt_ids, completion_ids, completion_logprobs, completion_mask=None):
    """Helper to create a TrajectoryStep with minimal boilerplate."""
    if completion_mask is None:
        completion_mask = [1] * len(completion_ids)
    return vf.TrajectoryStep(
        prompt=[],
        completion=[],
        response=MagicMock(),
        tokens=vf.TrajectoryStepTokens(
            prompt_ids=prompt_ids,
            prompt_mask=[0] * len(prompt_ids),
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_logprobs=completion_logprobs,
            overlong_prompt=False,
            is_truncated=False,
        ),
        reward=None,
        advantage=None,
        is_truncated=False,
        trajectory_id="test",
        extras={},
    )


def _make_state(trajectory, error=None):
    state = vf.State(trajectory=trajectory)
    if error is not None:
        state["error"] = error
    return state


def test_single_step():
    """Single trajectory step produces one sample, identical to old single-turn behavior."""
    state = _make_state([
        _make_step([1, 2], [3, 4], [-0.1, -0.2]),
    ])
    samples = interleave_rollout(state)

    assert samples is not None
    assert len(samples) == 1
    s = samples[0]
    assert s.prompt_ids == [1, 2]
    assert s.prompt_mask == [False, False]
    assert s.completion_ids == [3, 4]
    assert s.completion_mask == [True, True]
    assert s.completion_logprobs == [-0.1, -0.2]


def test_multi_step_extension_holds():
    """Two steps where extension holds — merged into one sample.
    Step 0: prompt=[1,2], completion=[3,4]
    Step 1: prompt=[1,2,3,4,5,6], completion=[7,8]
    Env feedback tokens are [5,6], model completion is [7,8]."""
    state = _make_state([
        _make_step([1, 2], [3, 4], [-0.1, -0.2]),
        _make_step([1, 2, 3, 4, 5, 6], [7, 8], [-0.3, -0.4]),
    ])
    samples = interleave_rollout(state)

    assert samples is not None
    assert len(samples) == 1
    s = samples[0]
    assert s.prompt_ids == [1, 2]
    assert s.completion_ids == [3, 4, 5, 6, 7, 8]
    assert s.completion_mask == [True, True, False, False, True, True]
    assert s.completion_logprobs == [-0.1, -0.2, 0.0, 0.0, -0.3, -0.4]


def test_three_steps_extension_holds():
    """Three steps all extending — one merged sample."""
    state = _make_state([
        _make_step([1, 2], [3, 4], [-0.1, -0.2]),
        _make_step([1, 2, 3, 4, 5, 6], [7, 8], [-0.3, -0.4]),
        _make_step([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12], [-0.5, -0.6]),
    ])
    samples = interleave_rollout(state)

    assert samples is not None
    assert len(samples) == 1
    s = samples[0]
    assert s.prompt_ids == [1, 2]
    assert s.completion_ids == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert s.completion_mask == [True, True, False, False, True, True, False, False, True, True]
    assert s.completion_logprobs == [-0.1, -0.2, 0.0, 0.0, -0.3, -0.4, 0.0, 0.0, -0.5, -0.6]


def test_extension_never_holds():
    """Two steps with completely different prefixes — two separate samples."""
    state = _make_state([
        _make_step([1, 2], [3, 4], [-0.1, -0.2]),
        _make_step([10, 20, 30, 40], [7, 8], [-0.3, -0.4]),
    ])
    samples = interleave_rollout(state)

    assert samples is not None
    assert len(samples) == 2

    assert samples[0].prompt_ids == [1, 2]
    assert samples[0].completion_ids == [3, 4]
    assert samples[0].completion_mask == [True, True]

    assert samples[1].prompt_ids == [10, 20, 30, 40]
    assert samples[1].completion_ids == [7, 8]
    assert samples[1].completion_mask == [True, True]


def test_extension_breaks_mid_trajectory():
    """Five steps: 1-3 extend, 4 breaks, 4-5 extend. Produces 2 samples."""
    state = _make_state([
        _make_step([1, 2], [3, 4], [-0.1, -0.2]),
        _make_step([1, 2, 3, 4, 5, 6], [7, 8], [-0.3, -0.4]),
        _make_step([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12], [-0.5, -0.6]),
        # Extension breaks — different prefix
        _make_step([100, 101, 102, 103], [104, 105], [-0.7, -0.8]),
        _make_step([100, 101, 102, 103, 104, 105, 106, 107], [108, 109], [-0.9, -1.0]),
    ])
    samples = interleave_rollout(state)

    assert samples is not None
    assert len(samples) == 2

    # Steps 1-3 merged
    s1 = samples[0]
    assert s1.prompt_ids == [1, 2]
    assert s1.completion_ids == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert s1.completion_mask == [True, True, False, False, True, True, False, False, True, True]
    assert s1.completion_logprobs == [-0.1, -0.2, 0.0, 0.0, -0.3, -0.4, 0.0, 0.0, -0.5, -0.6]

    # Steps 4-5 merged
    s2 = samples[1]
    assert s2.prompt_ids == [100, 101, 102, 103]
    assert s2.completion_ids == [104, 105, 106, 107, 108, 109]
    assert s2.completion_mask == [True, True, False, False, True, True]
    assert s2.completion_logprobs == [-0.7, -0.8, 0.0, 0.0, -0.9, -1.0]


def test_interleaved_agents():
    """Multi-prefix tracking: agent1 steps interleaved with agent2 step.
    agent1-step1, agent1-step2, agent2-step1, agent1-step3.
    agent1 steps merge together, agent2 is separate."""
    state = _make_state([
        _make_step([1, 2], [3, 4], [-0.1, -0.2]),
        _make_step([1, 2, 3, 4, 5, 6], [7, 8], [-0.3, -0.4]),
        # Agent2 — different prefix
        _make_step([100, 101], [102, 103], [-0.5, -0.6]),
        # Agent1 continues — extends agent1's prefix
        _make_step([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12], [-0.7, -0.8]),
    ])
    samples = interleave_rollout(state)

    assert samples is not None
    assert len(samples) == 2

    # Agent1: steps 0, 1, 3 merged
    a1 = samples[0]
    assert a1.prompt_ids == [1, 2]
    assert a1.completion_ids == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert a1.completion_mask == [True, True, False, False, True, True, False, False, True, True]
    assert a1.completion_logprobs == [-0.1, -0.2, 0.0, 0.0, -0.3, -0.4, 0.0, 0.0, -0.7, -0.8]

    # Agent2: step 2 alone
    a2 = samples[1]
    assert a2.prompt_ids == [100, 101]
    assert a2.completion_ids == [102, 103]
    assert a2.completion_mask == [True, True]
    assert a2.completion_logprobs == [-0.5, -0.6]


def test_empty_trajectory():
    state = _make_state([])
    assert interleave_rollout(state) is None


def test_error_rollout_masks_all_false():
    """When rollout has an error, all completion masks are False."""
    state = _make_state(
        [
            _make_step([1, 2], [3, 4], [-0.1, -0.2]),
            _make_step([1, 2, 3, 4, 5, 6], [7, 8], [-0.3, -0.4]),
        ],
        error="timeout",
    )
    samples = interleave_rollout(state)

    assert samples is not None
    assert len(samples) == 1
    s = samples[0]
    assert s.completion_ids == [3, 4, 5, 6, 7, 8]
    assert s.completion_mask == [False, False, False, False, False, False]
