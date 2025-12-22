# Testing the SFT config
# Run using uv run python tests/configs/test_sft_config.py @ configs/debug/sft/train.toml --max-steps=10


from nano_rl.trainer.sft.config import SFTTrainerConfig
from nano_rl.utils.pydantic_config import parse_argv


def main():
    config = parse_argv(SFTTrainerConfig)
    print(f"config={config}")


if __name__ == "__main__":
    main()
