# nano-rl

Repo for post training based on prime-RL
Some useful docs: https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/index.md

***Useful uv commands***
Install all dependencies from the lock file
uv sync --all-extras

Validate that your uv env is setup correctly
uv run python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

Kick off sft training on multiple gpus
uv run torchrun --nproc-per-node=8 -m nano_rl.trainer.sft.train @ configs/debug/sft/train.toml

single GPU run
uv run sft @ configs/debug/sft/train.toml

Before running vllm commands, run this, LD library path set
export LD_PRELOAD=/home/htkumar/nano_rl/.venv/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas.so.12

Starting inference server

CUDA_VISIBLE_DEVICES=0,1 uv run python -m nano_rl.inference.server @ configs/debug/infer.toml

Or just this
CUDA_VISIBLE_DEVICES=0,1 uv run inference  @ configs/debug/infer.toml

***Testing inference server***

curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "/home/htkumar/nano_rl/outputs/weights/step_100",
    "messages": [{"role": "user", "content": "capital of US!"}],
    "max_tokens": 50
  }'

***Run orchestrator***
First start the inference server and then run this
uv run orchestrator @ configs/debug/orch.toml

***Running RL***
Start inference server and then start orchestrator which writes the training batches

Then run this to start rl training
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 uv run torchrun --nproc_per_node=7 -m nano_rl.trainer.rl.train @ configs/debug/rl.toml

***Tests***
Running integration tests
uv run pytest tests/integration/test_vf.py -v -s
