# Tool Test

Train Qwen3-0.6B on the [tool-test](https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/tool_test) environment using RL (GRPO). This is a minimal example for validating tool-calling support through the RL pipeline.

## Environment

The tool-test environment is a single-turn tool-calling task. The model is given a random subset of 4 dummy tools (`tool_A`, `tool_B`, `tool_C`, `tool_D`) and must call exactly the requested set. Reward is binary: 1.0 if the correct tools are called, 0.0 otherwise.

## Setup

Install the environment:

```bash
prime env install primeintellect/tool-test
```

## RL Training

```bash
uv run rl @ configs/examples/tool_test/rl.toml
```

This starts inference (with `enable_auto_tool_choice` and `tool_call_parser = "hermes"`), orchestrator, and trainer on 2 GPUs.

## Evaluation

Start an inference server with tool calling enabled, then run eval:

```bash
CUDA_VISIBLE_DEVICES=0 uv run inference --model.name Qwen/Qwen3-0.6B --model.enable-auto-tool-choice true --model.tool-call-parser hermes

uv run python src/nano_rl/utils/vf_tool_env_inspect.py eval -n 20 -r 4
```

To evaluate a checkpoint, replace the model name:

```bash
CUDA_VISIBLE_DEVICES=0 uv run inference --model.name outputs/tool_test_rl/weights/step_10 --model.enable-auto-tool-choice true --model.tool-call-parser hermes

uv run python src/nano_rl/utils/vf_tool_env_inspect.py eval -n 20 -r 4 -m outputs/tool_test_rl/weights/step_10
```

To inspect a single rollout with detailed tool call / tool response output:

```bash
uv run python src/nano_rl/utils/vf_tool_env_inspect.py inspect
```

## Results

Qwen3-0.6B, 2 GPUs (1 inference + 1 trainer), NCCL weight broadcast.

| Stage | Avg Reward |
|-------|-----------|
| Base model | 0.725 |
| RL step 3 | 0.812 |
| RL step 10 | 0.863 |
