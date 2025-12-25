# nano-rl

Repo for post training based on prime-RL
Some useful docs: https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/index.md


***Useful uv commands***
Validate that your uv env is setup correctly
uv run python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

Kick off sft training on multiple gpus
uv run torchrun --nproc-per-node=8 src/nano_rl/trainer/sft/train.py @ configs/debug/sft/train.toml
