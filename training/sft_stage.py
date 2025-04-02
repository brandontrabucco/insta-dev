from datasets import load_from_disk

from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)

from typing import (
    Any,
    Dict,
    List,
    Tuple
)

import torch
import random

import argparse
import os


DEFAULT_DDP_TIMEOUT: int = 1e9


def preprocess_messages(
    messages: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer
) -> Tuple[List[int], List[int]]:
    """Creates masks for tokens to focus on assistant responses.

    Arguments:

    messages: List of chat messages.
    tokenizer: Tokenizer to use for encoding.

    Returns:

    input_ids: List of token IDs.
    mask: List of mask values.

    """

    input_ids: List[int] = []
    mask: List[int] = []

    for chat_turn in messages:
        
        chat_turn_text = tokenizer.apply_chat_template(
            [chat_turn],
            tokenize = False,
            add_generation_prompt = False
        )
        
        chat_turn_tokens = tokenizer.encode(
            chat_turn_text,
            add_special_tokens = False
        )

        chat_turn_mask = [
            1 if chat_turn["role"] == "assistant" else 0
        ] * len(chat_turn_tokens)

        input_ids.extend(chat_turn_tokens)
        mask.extend(chat_turn_mask)

    need_bos_token = (
        tokenizer.bos_token_id is not None
        and tokenizer.bos_token_id not in input_ids[:2]
    )

    if need_bos_token:

        input_ids = [
            tokenizer.bos_token_id
        ] + input_ids

        mask = [0] + mask

    need_eos_token = (
        tokenizer.eos_token_id is not None
        and tokenizer.eos_token_id not in input_ids[-2:]
    )

    if need_eos_token:

        input_ids = input_ids + [
            tokenizer.eos_token_id
        ]

        mask = mask + [1]

    return input_ids, mask


class ChatDataCollator:

    def __init__(
        self, tokenizer: PreTrainedTokenizer = None,
        max_seq_length: int = 8192
    ):

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Collate a batch of messages into a tensor of input ids,
        and a mask that indicates which tokens are assistant responses.

        Arguments:

        features: List of chat messages.

        Returns:

        batch: Dictionary containing input IDs, attention mask, and labels.
        
        """
    
        # Extract messages from each example
        batch_messages = [
            feature['messages']
            for feature in features
        ]
        
        # Process each example
        batch_tokens = []
        batch_attention_masks = []
        batch_labels = []
        
        for messages in batch_messages:

            tokens, loss_mask = preprocess_messages(
                messages = messages,
                tokenizer = self.tokenizer
            )

            tokens = tokens[:self.max_seq_length]
            loss_mask = loss_mask[:self.max_seq_length]

            batch_tokens.append(tokens)
            attention_mask = [1] * len(tokens)
            batch_attention_masks.append(attention_mask)

            labels = [
                -100 if mask == 0 else token
                for token, mask in zip(tokens, loss_mask)
            ]

            batch_labels.append(labels)
            
        for batch_idx in range(len(batch_tokens)):

            padding_length = (
                self.max_seq_length - 
                len(batch_tokens[batch_idx])
            )

            if padding_length == 0:

                continue

            batch_attention_masks[batch_idx].extend(
                [0] * padding_length
            )

            batch_labels[batch_idx].extend(
                [-100] * padding_length
            )

            batch_tokens[batch_idx].extend([
                self.tokenizer.pad_token_id
            ] * padding_length)
        
        attention_mask_tensor = torch.tensor(batch_attention_masks)
        batch_tokens = torch.tensor(batch_tokens)
        batch_labels = torch.tensor(batch_labels)

        return {
            "input_ids": batch_tokens,
            "attention_mask": attention_mask_tensor,
            "labels": batch_labels
        }


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
        default="./insta-150k-v2-filtered"
    )

    parser.add_argument(
        "--output_dir",
        type = str,
        default="./qwen-1.5b-filtered"
    )

    parser.add_argument(
        "--final_model_dir",
        type = str,
        default="./qwen-1.5b-filtered"
    )

    parser.add_argument(
        "--max_seq_length",
        type = int,
        default = 8192
    )

    parser.add_argument(
        "--max_num_examples",
        type = int,
        default = None
    )

    parser.add_argument(
        "--use_bf16",
        action = "store_true"
    )

    args = parser.parse_args()

    insta_dataset = load_from_disk(
        args.dataset_path
    )

    if args.max_num_examples is not None:

        insta_dataset = insta_dataset.select(
            random.Random(0).sample(
                list(range(len(insta_dataset))),
                args.max_num_examples
            )
        )

    training_args = TrainingArguments(
        ddp_timeout = DEFAULT_DDP_TIMEOUT,
        optim = "adamw_torch_fused",
        gradient_accumulation_steps = 4,
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = {
            'use_reentrant': False
        },
        learning_rate = 5e-5,
        weight_decay = 0,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-8,
        num_train_epochs = 1,
        warmup_ratio = 0.01,
        logging_steps = 100,
        output_dir = args.output_dir,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        bf16 = args.use_bf16,
        remove_unused_columns = False,
        save_total_limit = 3,
        save_steps = 1000,
        save_only_model = True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype = torch.bfloat16,
        attn_implementation = "flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    data_collator = ChatDataCollator(
        tokenizer = tokenizer,
        max_seq_length = args.max_seq_length
    )

    trainer = Trainer(
        model = model,
        train_dataset = insta_dataset,
        args = training_args,
        data_collator = data_collator
    )

    trainer.train()

    trainer.save_model(
        args.final_model_dir
    )

    tokenizer.save_pretrained(
        args.final_model_dir
    )