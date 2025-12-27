# nano-rl

Repo for post training based on prime-RL
Some useful docs: https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/index.md

***Useful uv commands***
Install all dependencies from the lock file
uv sync --all-extras

Validate that your uv env is setup correctly
uv run python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

Kick off sft training on multiple gpus
uv run torchrun --nproc-per-node=8 src/nano_rl/trainer/sft/train.py @ configs/debug/sft/train.toml

Before running vllm commands, run this, LD library path set
export LD_PRELOAD=/home/htkumar/nano_rl/.venv/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas.so.12

Starting inference server

CUDA_VISIBLE_DEVICES=0,1 uv run python -m nano_rl.inference.server @ configs/debug/infer.toml

Testing inference server

curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "/home/htkumar/nano_rl/outputs/weights/step_100",
    "messages": [{"role": "user", "content": "capital of US!"}],
    "max_tokens": 50
  }'
