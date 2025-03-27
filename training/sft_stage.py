from datasets import load_from_disk

from trl import (
    SFTConfig,
    SFTTrainer
)

import argparse
import os


DEFAULT_DDP_TIMEOUT: int = 1e9


if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type = str,
        default="Qwen/Qwen2.5-1.5B-Instruct"
    )

    parser.add_argument(
        "--dataset_path",
        type = str,
        default="./insta-150k-v2"
    )

    parser.add_argument(
        "--output_dir",
        type = str,
        default="./qwen-1.5b"
    )

    parser.add_argument(
        "--max_seq_length",
        type = int,
        default = 8196
    )

    parser.add_argument(
        "--use_bf16",
        action = "store_true"
    )

    args = parser.parse_args()

    insta_dataset = load_from_disk(
        args.dataset_path
    )

    training_args = SFTConfig(
        ddp_timeout = DEFAULT_DDP_TIMEOUT,
        optim = "adamw_torch_fused",
        gradient_accumulation_steps = 4,
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = {'use_reentrant': False},
        model_init_kwargs = {
            "attn_implementation": "flash_attention_2",
            "torch_dtype": "bfloat16",
        },
        learning_rate = 1e-5,
        num_train_epochs = 1,
        max_length = args.max_seq_length,
        output_dir = args.output_dir,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        bf16 = args.use_bf16,
    )

    trainer = SFTTrainer(
        model = args.model_name,
        train_dataset = insta_dataset,
        args = training_args,
    )

    trainer.train()