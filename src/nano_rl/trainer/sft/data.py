"""
Data loader for SFT
Assumes training is run on a single GPU
"""

from collections import defaultdict
from typing import cast, Literal, TypedDict

import torch
from datasets import Dataset, load_dataset
from jaxtyping import Bool, Int

from nano_rl.trainer.sft.config import DataConfigType, LossMaskConfig
from nano_rl.trainer.world import get_world
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer


# one training sample
class Sample(TypedDict):
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[bool]
    target_ids: list[int]


# batch of samples of shape (batch_size, seq_len)
class Batch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]
    target_ids: Int[Tensor, "batch seq"]


class StatefulIterableDataset(Stateful, IterableDataset):
    """
    Base class with checkpointing via PyTorch DCP stateful interface and iterable dataset
    """

    def __init__(self):
        self.step, self.epoch = 0, 0
        self._fast_forward = False
        # num_samples and num_tokens for different dataset splits (train, test etc.)
        self.num_samples: dict[str, int] = defaultdict(int)
        self.num_tokens: dict[str, int] = defaultdict(int)

    def state_dict(self) -> dict:
        return {"step": self.step, "epoch": self.epoch}

    def load_state_dict(self, state_dict: dict):
        assert "step" in state_dict and "epoch" in state_dict
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]
        self._fast_forward = True


class FakeDataset(StatefulIterableDataset):
    """Fake dataset for debugging"""

    def __init__(self, vocab_size, seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def __iter__(self):
        while True:
            self.step += 1
            input_ids = torch.randint(0, self.vocab_size, (self.seq_len + 1,)).tolist()

            yield {
                "input_ids": input_ids[:-1],
                "target_ids": input_ids[1:],
                "position_ids": list(range(self.seq_len)),
                "loss_mask": [True] * self.seq_len,
            }


class SFTDataset(StatefulIterableDataset):
    """
    HF datasets which have prompt+completion format like willcb/R1-reverse-wikipedia-paragraphs-v1-1000
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer | None,
        shuffle: bool = True,
        seed: int = 0,
        seq_len: int = 128,
        loss_mask_config: LossMaskConfig = LossMaskConfig(),
        max_examples: int | None = None,
        max_epochs: int | None = None,
        non_dp_size: int = 1,
    ):
        super().__init__()
        self.dataset = dataset
        self.num_examples = len(dataset)
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.seed = seed
        self.seq_len = seq_len
        self.loss_mask_config = loss_mask_config
        self.max_examples = max_examples
        self.max_epochs = max_epochs

        if max_examples is not None:
            self.num_examples = min(self.num_examples, max_examples)
            self.dataset = dataset.take(max_examples)

        world = get_world()
        assert (
            world.world_size % non_dp_size == 0
        ), "World size should be a multiple of non_sp_size"
        self.data_rank = world.rank // non_dp_size
        # number of groups of GPUs that see different data during training
        self.data_world_size = world.world_size // non_dp_size

    def _should_mask(self, message: dict) -> bool:
        """Check if message role contributes to the loss"""
        role = message["role"]
        match role:
            case "user":
                return self.loss_mask_config.user
            case "system":
                return self.loss_mask_config.system
            case "assistant":
                return self.loss_mask_config.assistant
            case "tool":
                return self.loss_mask_config.tool
            case _:
                raise ValueError(f"Invalid role: {role}")

    def _build_loss_mask(
        self, prompt: list[dict], completion: list[dict], tools: list, kwargs: dict
    ):
        messages = prompt + completion
        loss_mask: list[bool] = []
        prev_ids: list[int] = []
        prev_len = 0

        for i, msg in enumerate(messages):
            # Handling parallel tool calls as some chat templates treat this as one message
            if (
                msg["role"] == "tool"
                and i + 1 < len(messages)
                and messages[i + 1]["role"] == "tool"
            ):
                continue

            # Add generation prompt after user/tool before assistant like <im_start> assistant. We don't want keep_loss True for these tokens
            add_gen_prompt = (
                msg["role"] in ["user", "tool"]
                and i + 1 < len(messages)
                and messages[i + 1]["role"] == "assistant"
            )
            cur_ids = self.tokenizer.apply_chat_template(
                messages[: i + 1],
                tools=tools,
                add_generation_prompt=add_gen_prompt,
                return_dict=False,
                **kwargs,
            )
            # if prev_ids != cur_ids[:prev_len]:
            #     print(f"i={i}, role={msg['role']}")
            #     print(f"prev_ids ({len(prev_ids)}): {self.tokenizer.decode(prev_ids)}")
            #     print(
            #         f"cur_ids[:prev_len] ({prev_len}): {self.tokenizer.decode(cur_ids[:prev_len])}"
            #     )
            #     print(f"Full cur_ids decoded: {self.tokenizer.decode(cur_ids)}")

            # We assume that the chat template we are using satisfies this constraint
            assert prev_ids == cur_ids[:prev_len], f"Tokenization mismatch at msg {i}"
            loss_mask.extend([self._should_mask(msg)] * (len(cur_ids) - prev_len))
            prev_len, prev_ids = len(cur_ids), cur_ids

        return loss_mask

    def _process(self, example: dict) -> Sample | None:
        """
        Process HF dataset example into training format
        TODO: Handle tools in examples
        """
        if self.tokenizer is None:
            # Assume pre-tokenized - validate required keys exist
            required = {"input_ids", "target_ids", "loss_mask", "position_ids"}
            if not required.issubset(example.keys()):
                raise ValueError(
                    f"Pre-tokenized example missing keys: {required - example.keys()}"
                )
            return example

        if "prompt" not in example or "completion" not in example:
            raise ValueError("Example needs prompt and completionn cols")

        def strip_content(messages: list[dict]) -> list[dict]:
            return [
                (
                    {**m, "content": m["content"].strip()}
                    if isinstance(m.get("content"), str)
                    else m
                )
                for m in messages
            ]

        prompt = strip_content(example["prompt"])
        completion = strip_content(example["completion"])
        kwargs = example.get("chat_template_kwargs", {})
        input_ids = self.tokenizer.apply_chat_template(
            prompt + completion,
            tools=[],
            return_dict=False,
            **kwargs,
        )
        loss_mask = self._build_loss_mask(prompt, completion, [], kwargs)

        if self.tokenizer.eos_token_id not in input_ids:
            input_ids.append(self.tokenizer.eos_token_id)
            loss_mask.append(True)

        target_ids = input_ids[1:]
        input_ids = input_ids[:-1]
        loss_mask = loss_mask[1:]
        if sum(loss_mask[: self.seq_len]) == 0:
            return None

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "position_ids": list(range(len(input_ids))),
        }

    def __iter__(self):
        dataset = (
            self.dataset.shuffle(seed=self.epoch + self.seed)
            if self.shuffle
            else self.dataset
        )
        while True:
            self.step += 1
            epoch = (self.step - 1) // self.num_examples
            if self.max_epochs is not None and epoch >= self.max_epochs:
                break

            if epoch > self.epoch:
                self.epoch = epoch
                dataset = (
                    self.dataset.shuffle(seed=self.epoch + self.seed)
                    if self.shuffle
                    else self.dataset
                )

            # this gpu shouldn't process this sample.
            if (self.step - 1) % self.data_world_size != self.data_rank:
                continue

            example = dataset[(self.step - 1) % self.num_examples]
            processed = self._process(example)
            if processed:
                yield processed


class CatDataset(StatefulIterableDataset):
    """Concatenates samples to fixed seq_len packing"""

    def __init__(self, dataset: StatefulIterableDataset, seq_len: int):
        super().__init__()
        self.dataset = dataset
        self.seq_len = seq_len

    def state_dict(self) -> dict:
        return {"dataset": self.dataset.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.dataset.load_state_dict(state_dict["dataset"])

    def __iter__(self):
        packed: dict[str, list] = defaultdict(list)
        cur_len = 0
        for sample in self.dataset:
            for key, value in sample.items():
                packed[key].extend(value)
            cur_len += len(sample["input_ids"])

            if cur_len >= self.seq_len:
                yield {k: v[: self.seq_len] for k, v in packed.items()}
                # reset buffers once desired seq_len is reached. Some data loss here, but it is fine.
                packed = defaultdict(list)
                cur_len = 0


def cat_collate_fn(samples: list[Sample]) -> Batch:
    """Collate for cat packed samples"""
    return {
        "input_ids": torch.stack([torch.tensor(s["input_ids"]) for s in samples])
        .long()
        .cuda(),
        "target_ids": torch.stack([torch.tensor(s["target_ids"]) for s in samples])
        .long()
        .cuda(),
        "position_ids": torch.stack([torch.tensor(s["position_ids"]) for s in samples])
        .long()
        .cuda(),
        "loss_mask": torch.stack([torch.tensor(s["loss_mask"]) for s in samples])
        .bool()
        .cuda(),
    }


def setup_dataset(
    tokenizer: PreTrainedTokenizer, config: DataConfigType, non_dp_size: int = 1
) -> StatefulIterableDataset:
    if config.type == "fake":
        return FakeDataset(tokenizer.vocab_size, config.seq_len)

    dataset = load_dataset(config.name, split="train")
    return SFTDataset(
        dataset,
        tokenizer,
        shuffle=config.shuffle,
        seed=config.seed,
        seq_len=config.seq_len,
        loss_mask_config=config.loss_mask,
        non_dp_size=non_dp_size,
    )


def setup_dataloader(
    dataset: StatefulIterableDataset, config: DataConfigType
) -> StatefulDataLoader:
    packed = CatDataset(dataset, config.seq_len * config.micro_batch_size)
    # Cat dataset creates a dataset of dim (seq_len * micro_batch_size).
    # When we create stateful data loader, we always use batch_size 1 as we already have all the tokens.
    return StatefulDataLoader(packed, batch_size=1, collate_fn=cat_collate_fn)


if __name__ == "__main__":
    from nano_rl.trainer.sft.config import SFTDataConfig
    from transformers import AutoTokenizer

    print("Testing SFTDataset with willcb/R1-reverse-wikipedia-paragraphs-v1-1000")

    # Load tokenizer (same way as prime-rl)
    tokenizer = AutoTokenizer.from_pretrained(
        "PrimeIntellect/Qwen3-0.6B", trust_remote_code=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    config = SFTDataConfig(
        name="willcb/R1-reverse-wikipedia-paragraphs-v1-1000",
        seq_len=4096,
        batch_size=32,
        micro_batch_size=1,
        shuffle=False,
    )

    dataset = setup_dataset(tokenizer, config)
    sample = next(iter(dataset))
    print(f"Sample input_ids length: {len(sample['input_ids'])}")
    print(f"Tokens with loss: {sum(sample['loss_mask'])} / {len(sample['loss_mask'])}")

    dataloader = setup_dataloader(dataset, config)
    batch = next(iter(dataloader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch target_ids shape: {batch['target_ids'].shape}")
    print(f"Batch loss_mask shape: {batch['loss_mask'].shape}")
    print(f"Batch device: {batch['input_ids'].device}")

    # Verify loss masking
    total_tokens = batch["loss_mask"].numel()
    tokens_with_loss = batch["loss_mask"].sum().item()
    print(f"Tokens with loss in batch: {tokens_with_loss} / {total_tokens}")
