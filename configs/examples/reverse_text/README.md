# Reverse Text

Train Qwen3-0.6B to reverse text using SFT followed by RL (GRPO).

## Pipeline

### 1. SFT Training

```bash
uv run torchrun --nproc-per-node=8 -m nano_rl.trainer.sft.train @ configs/examples/reverse_text/sft/train.toml
```

### 2. RL Training

Using the unified launcher (starts inference, orchestrator, and trainer):

```bash
uv run rl @ configs/examples/reverse_text/rl/rl.toml
```

Or manually start each component separately:

```bash
# Start inference servers (4 GPUs)
CUDA_VISIBLE_DEVICES=0 uv run inference @ configs/examples/reverse_text/rl/infer.toml --port 8000
CUDA_VISIBLE_DEVICES=1 uv run inference @ configs/examples/reverse_text/rl/infer.toml --port 8001
CUDA_VISIBLE_DEVICES=2 uv run inference @ configs/examples/reverse_text/rl/infer.toml --port 8002
CUDA_VISIBLE_DEVICES=3 uv run inference @ configs/examples/reverse_text/rl/infer.toml --port 8003

# Start orchestrator
uv run orchestrator @ configs/examples/reverse_text/rl/orch.toml

# Start trainer (4 GPUs)
CUDA_VISIBLE_DEVICES=4,5,6,7 uv run torchrun --nproc_per_node=4 -m nano_rl.trainer.rl.train @ configs/examples/reverse_text/rl/train.toml
```

Weights are saved at `outputs/rl/weights/step_{n}`.

### 3. Evaluation

Start an inference server with a checkpoint, then run eval:

```bash
CUDA_VISIBLE_DEVICES=0 uv run inference --model.name outputs/rl/weights/step_100
uv run vf-eval reverse-text -m outputs/rl/weights/step_100 -b http://localhost:8000/v1 -n 20 --max-tokens 1024
```

## Results

### SFT Benchmarks (Qwen3-0.6B, batch_size=192, seq_len=4096)

| Config | MFU | Throughput | Time/Step | Peak Memory |
|--------|-----|------------|-----------|-------------|
| torch.compile | 40.74% | 228.17K tok/s | 3.15s | 44.2 GiB (55.9%) |
| torch.compile + AC | 21.76% | 121.85K tok/s | 6.46s | — |

Activation checkpointing trades compute for memory. Not recommended for Qwen3-0.6B on A100 since memory is not the bottleneck.

### RL Reward Progression

| Stage | Avg Reward |
|-------|-----------|
| Base model (Qwen3-0.6B) | 0.063 |
| After SFT | 0.535 |
| RL step 0 | 0.545 |
| RL step 10 | 0.718 |
| RL step 20 | 0.781 |
| RL step 80 | 0.802 |
| RL step 100 | 0.814 |

RL step 0 is close to SFT as expected (same weights, just starting RL). Clear improvement through RL training, with most gains in the first 20 steps.
