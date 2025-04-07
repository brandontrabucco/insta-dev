from datasets import load_from_disk

from insta import (
    ACTION_PARSERS,
    BaseActionParser,
    BrowserStatus,
)

from insta.utils import safe_call

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from trl import (
    GRPOConfig,
    GRPOTrainer,
)

from functools import partial
from typing import List

import json
import torch
import random

import editdistance
import argparse
import os


DEFAULT_DDP_TIMEOUT: int = 1e9


def action_reward_function(
    prompts: List[str],
    completions: List[str],
    output: List[str],
    success: List[float],
    action_parser: BaseActionParser = None
) -> List[float]:
    """Reward function for training LLM agents to operate a browser,
    and complete a desired web navigation task.

    Arguments:

    prompts: List[str]
        List of prompts used to generate completions.

    completions: List[str]
        List of completions generated by the LLM.

    output: List[str]
        List of outputs generated by the LLM.

    success: List[float]
        List of success values for the original trajectories.

    action_parser: BaseActionParser
        Action parser used that is used to determine if the completion
        is valid, and to extract the final action.

    Returns:

    rewards: List[float]
        List of rewards for each completion.
    
    """
    
    rewards: List[float] = []

    for prompt, completion, ground_truth, original_success in zip(
        prompts, completions, output, success
    ):

        ground_truth = json.loads(
            ground_truth
        )

        if isinstance(completion, list):

            completion = completion[0]['content']

        action = safe_call(
            action_parser.parse_action,
            response = completion,
            catch_errors = True,
            max_errors = 1,
            log_errors = False
        )

        if action is BrowserStatus.ERROR:

            rewards.append(
                -1.0 * original_success
            )

            continue

        action = json.loads(
            action.matched_response
        )

        has_required_keys = (
            "action_key" in action and
            "target_element_id" in action and
            "action_kwargs" in action
        )

        if not has_required_keys:

            rewards.append(
                -1.0 * original_success
            )

            continue

        reward = 0.1

        action_key_match = (
            ground_truth["action_key"] == 
            action["action_key"]
        )

        if action_key_match:

            reward += 0.3

        target_element_match = (
            ground_truth["target_element_id"] == 
            action["target_element_id"]
        )

        if target_element_match:

            reward += 0.3

        ground_truth_kwargs_str = json.dumps(
            ground_truth["action_kwargs"]
        )

        action_kwargs_str = json.dumps(
            action["action_kwargs"]
        )

        action_kwargs_match = (
            ground_truth_kwargs_str ==
            action_kwargs_str
        )

        if action_kwargs_match:

            reward += 0.3

        rewards.append(
            reward
        )

    return rewards


if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "8"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type = str,
        default = "./qwen-1.5b-filtered"
    )

    parser.add_argument(
        "--dataset_path",
        type = str,
        default = "./insta-150k-v2-grpo-round=0"
    )

    parser.add_argument(
        "--output_dir",
        type = str,
        default = "./qwen-1.5b-grpo-round=0"
    )

    parser.add_argument(
        "--action_parser",
        type = str,
        default = "simplified_json"
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

    action_parser = ACTION_PARSERS[
        args.action_parser
    ]()

    action_reward_function = partial(
        action_reward_function,
        action_parser = action_parser
    )

    action_reward_function.__name__ = "action_reward_function"

    if args.max_num_examples is not None:

        insta_dataset = insta_dataset.select(
            random.Random(0).sample(
                list(range(len(insta_dataset))),
                args.max_num_examples
            )
        )

    # ssh -N -L 8000, 51216

    training_args = GRPOConfig(
        ddp_timeout = DEFAULT_DDP_TIMEOUT,
        optim = "adamw_torch_fused",
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,
        gradient_accumulation_steps = 8,
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = {
            'use_reentrant': False
        },
        learning_rate = 5e-5,
        weight_decay = 0.0,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        adam_epsilon = 1e-8,
        max_prompt_length = 7680,
        max_completion_length = 512,
        temperature = 0.7,
        top_k = None,
        top_p = 1,
        num_generations = 8,
        num_iterations = 1,
        beta = 0.04,
        epsilon = 0.2,
        num_train_epochs = 10,
        warmup_steps = 0,
        logging_steps = 1,
        output_dir = args.output_dir,
        bf16 = args.use_bf16,
        remove_unused_columns = False,
        save_total_limit = 3,
        save_steps = 1000,
        save_only_model = True,
        use_vllm = True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype = torch.bfloat16,
        attn_implementation = "flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    trainer = GRPOTrainer(
        model = model,
        train_dataset = insta_dataset,
        reward_funcs = action_reward_function,
        args = training_args,
    )

    trainer.train()

    tokenizer.save_pretrained(
        args.output_dir
    )

    trainer.save_model(
        args.output_dir
    )