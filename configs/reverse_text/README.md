For this task, we do SFT and RL

To start inferece server with a model name without toml config, use this
CUDA_VISIBLE_DEVICES=0 uv run inference --model.name outputs/run_r/weights/step_100


On base QWEN model, we run eval using
uv run vf-eval reverse-text -m PrimeIntellect/Qwen3-0.6B -b http://localhost:8000/v1 -n 20 --ma
x-tokens 1024

avg reward we get is 0.063

With SFT model, we run eval using
uv run vf-eval reverse-text -m outputs/weights/step_100 -b http://localhost:8000/v1 -n 20 --max-tokens 1024

avg reward we get is 0.535

Now do RL training, weights are saved at outputs/run_r/weights/step_{n}

At step 20, we run
uv run vf-eval reverse-text -m outputs/run_r/weights/step_20 -b http://localhost:8000/v1 -n 20 --max-tokens 1024

and get average reward of 0.781

Step 0 reward is 0.545 which is very close to SFT as expected.
Step 10 we see 0.718 which is in between

Step 80 we get 0.802
Step 100 we get 0.814
