"""
SFT trainer
"""

import time

import torch
import torch.nn as nn
from nano_rl.trainer.model import setup_model, setup_tokenizer
from nano_rl.trainer.optim import setup_optimizer
from nano_rl.trainer.scheduler import setup_scheduler
from nano_rl.trainer.sft.config import SFTTrainerConfig
from nano_rl.trainer.sft.data import setup_dataloader, setup_dataset
from nano_rl.utils.pydantic_config import parse_argv
from torch.nn import CrossEntropyLoss


def train(config: SFTTrainerConfig):
    device = torch.device("cuda")

    # Model & tokenizer
    print(f"Loading model: {config.model.name}")
    model = setup_model(config.model).to(device)

    print(f"Loading tokenizer: {config.tokenizer.name}")
    tokenizer = setup_tokenizer(config.tokenizer)

    # Optimizer & scheduler
    optimizer = setup_optimizer(config.optim, model)
    scheduler = setup_scheduler(
        optimizer, config.scheduler, config.max_steps, config.optim.lr
    )

    # Data
    print(f"Loading dataset")
    dataset = setup_dataset(tokenizer, config.data)
    dataloader = setup_dataloader(dataset, config.data)
    dataiter = iter(dataloader)

    # Gradient accumulation
    grad_accum_steps = config.data.batch_size // config.data.micro_batch_size

    # loss function
    ce_loss = CrossEntropyLoss(reduction="none")
    print(f"Starting training (max_steps={config.max_steps})")
    step = 0

    while step < config.max_steps:
        start_time = time.perf_counter()
        batch_loss = 0.0
        optimizer.zero_grad()

        for _ in range(grad_accum_steps):
            batch = next(dataiter)
            input_ids = batch["input_ids"].to(device)
            position_ids = batch["position_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            # forward pass
            logits = model(input_ids=input_ids, position_ids=position_ids).logits
            B, L, V = logits.shape

            # Loss with masking
            loss = ce_loss(logits.view(-1, V), target_ids.view(-1)).view(B, L)
            loss = loss[loss_mask].mean()

            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()
            batch_loss += scaled_loss.item()

        # Does gradient clipping in place
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # total_norm = sqrt(
        #     sum(x.grad**2 for x in model.parameters() if x.grad is not None)
        # )
        optimizer.step()
        scheduler.step()
        step_time = time.perf_counter() - start_time
        print(
            f"step {step} | Loss {batch_loss:.4f} | Grad Norm {grad_norm:.4f} | Step Time {step_time:.2}"
        )
        step += 1


print("Training done")


def main():
    config = parse_argv(SFTTrainerConfig)
    train(config)


if __name__ == "__main__":
    main()
