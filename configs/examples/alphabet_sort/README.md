# Alphabet Sort

Multi-turn RL example training `Qwen/Qwen3-4B-Instruct-2507` to sort names alphabetically using full fine-tuning. This task doesn't require SFT warmup as the base model already understands the conversation format.

> This example runs on 8 GPUs (4 inference, 4 training).

## Setup

Install the environment:
```bash
prime env install primeintellect/alphabet-sort
```

Verify installation:
```bash
uv run python -c "import alphabet_sort"
```

## Task

This multi-turn conversation task requires the model to:
- Sort names alphabetically by first OR last name (randomly chosen per episode)
- Maintain a cumulative sorted list across multiple turns
- Tag new names with `// new name!` marker
- Handle several names per turn

## Baseline Evaluation

Start the inference server:
```bash
CUDA_VISIBLE_DEVICES=0 uv run inference --model.name Qwen/Qwen3-4B-Instruct-2507
```

Evaluate the base model:
```bash
uv run vf-eval alphabet-sort \
  -m Qwen/Qwen3-4B-Instruct-2507 \
  -b http://localhost:8000/v1 \
  -n 20 \
  --max-tokens 768 \
  --env-args '{"min_turns": 3, "max_turns": 3, "min_names_per_turn": 1, "max_names_per_turn": 4, "similarity_power": 8, "power_per_turn": false}'
```

Baseline results (avg reward ~0.265):
```
reward: avg - 0.265, std - 0.243
r1: [0.059, 0.264, 0.731, 0.23, 0.103, 0.014, 0.099, 0.115, 0.199, 0.037, 0.181, 0.014, 0.028, 0.834, 0.207, 0.625, 0.269, 0.206, 0.501, 0.558]
r2: [0.059, 0.264, 0.731, 0.23, 0.132, 0.014, 0.099, 0.115, 0.199, 0.037, 0.181, 0.014, 0.028, 0.834, 0.207, 0.625, 0.269, 0.206, 0.501, 0.558]
r3: [0.059, 0.264, 0.731, 0.23, 0.132, 0.014, 0.099, 0.115, 0.199, 0.037, 0.181, 0.014, 0.028, 0.834, 0.207, 0.625, 0.269, 0.206, 0.501, 0.558]
```

## RL Training

We train with full fine-tuning (no LoRA) for 100 steps with `lr=5e-7`, `batch_size=96`, `rollouts_per_example=8`, and `micro_batch_size=8`.

```bash
uv run rl @ configs/examples/alphabet_sort/rl.toml
```

## Post-RL Evaluation

Load the trained checkpoint on the inference server:
```bash
CUDA_VISIBLE_DEVICES=0 uv run inference --model.name outputs/runs/alphabet_sort_rl/weights/step_100
```

Evaluate:
```bash
uv run vf-eval alphabet-sort \
  -m outputs/runs/alphabet_sort_rl/weights/step_100 \
  -b http://localhost:8000/v1 \
  -n 20 \
  --max-tokens 768 \
  --env-args '{"min_turns": 3, "max_turns": 3, "min_names_per_turn": 1, "max_names_per_turn": 4, "similarity_power": 8, "power_per_turn": false}'
```

Post-RL results (avg reward ~0.764):
```
reward: avg - 0.764, std - 0.321
r1: [1.0, 0.775, 1.0, 1.0, 1.0, 1.0, 0.718, 1.0, 0.543, 0.416, 1.0, 0.243, 0.007, 1.0, 1.0, 0.484, 1.0, 1.0, 1.0, 0.698]
r2: [1.0, 0.775, 1.0, 1.0, 0.175, 1.0, 0.032, 1.0, 0.812, 0.77, 1.0, 0.243, 0.402, 1.0, 1.0, 0.289, 1.0, 1.0, 1.0, 0.698]
r3: [1.0, 0.775, 1.0, 1.0, 0.175, 0.633, 0.032, 1.0, 0.812, 0.416, 1.0, 0.2, 0.384, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.351]
```

## Notes

- **Full fine-tuning**: This example uses full fine-tuning (no LoRA) with `lr=5e-7`. Higher learning rates (e.g., `1e-5`) cause mode collapse with full fine-tuning.
- **Batch size**: We use `batch_size=96` and `rollouts_per_example=8`. Larger batch sizes would likely improve results further.
