# nano-rl

Minimal RL post-training framework based on [prime-RL](https://github.com/PrimeIntellect-ai/prime-rl).

## Setup

Install all dependencies:
```bash
uv sync --all-extras
```

Validate your environment:
```bash
uv run python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

Before running vLLM commands, set the LD library path:
```bash
export LD_PRELOAD=/home/htkumar/nano_rl/.venv/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas.so.12
```

## SFT Training

Single GPU:
```bash
uv run sft @ configs/debug/sft/train.toml
```

Multi-GPU:
```bash
uv run torchrun --nproc-per-node=8 -m nano_rl.trainer.sft.train @ configs/debug/sft/train.toml
```

## RL Training

RL training requires three components running in parallel:

### 1. Start Inference Server(s)

```bash
CUDA_VISIBLE_DEVICES=0 uv run inference @ configs/debug/infer.toml --port 8000
CUDA_VISIBLE_DEVICES=1 uv run inference @ configs/debug/infer.toml --port 8001
```

### 2. Start Orchestrator

```bash
uv run orchestrator @ configs/debug/orch.toml
```

### 3. Start RL Trainer

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 uv run torchrun --nproc_per_node=6 -m nano_rl.trainer.rl.train @ configs/debug/rl.toml
```

## Evaluation

Run one-off evaluations against a verifiers environment:
```bash
uv run vf-eval reverse-text -m Qwen/Qwen3-0.6B -b http://localhost:8000/v1 -n 20 --max-tokens 1024
```

## Testing

Run integration tests:
```bash
uv run pytest tests/integration/test_vf.py -v -s
```

## Testing Inference Server

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "capital of US?"}],
    "max_tokens": 50
  }'
```

## Resources

- [prime-RL documentation](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/index.md)

## Backlog

- [ ] NCCL broadcast of weights to inference servers
- [ ] Unified `rl.py` launcher to run full RL loop with one command
