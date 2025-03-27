from datasets import load_dataset

from insta.configs.agent_config import (
    AgentConfig,
    get_agent_config,
    DEFAULT_AGENT_CONFIG
)

from insta.agent import BrowserAgent
from functools import partial

import argparse
import json
import os


def select_valid_samples(
    example: dict,
    base_data_dir: str = "data-v3",
    success_threshold: float = 0.9,
    future_success_threshold: float = 0.9,
    reasoning_is_correct_threshold: float = 0.9
) -> bool:

    observations_dir = os.path.join(
        base_data_dir,
        "observations"
    )

    actions_dir = os.path.join(
        base_data_dir,
        "actions"
    )

    judgments_dir = os.path.join(
        base_data_dir,
        "judgments"
    )
            
    valid_domain = (
        os.path.exists(
            os.path.join(
                observations_dir,
                "{}.json".format(example["domain"])
            )
        ) and os.path.exists(
            os.path.join(
                actions_dir,
                "{}.json".format(example["domain"])
            )
        ) and os.path.exists(
            os.path.join(
                judgments_dir,
                "{}.json".format(example["domain"])
            )
        )
    )

    if not valid_domain:

        return False
    
    judgments_path = os.path.join(
        judgments_dir,
        "{}.json".format(example["domain"])
    )     

    try:  # file may be corrupted  

        with open(judgments_path, "r") as file:
            judgments = json.load(file)

    except json.decoder.JSONDecodeError:

        return False

    success = judgments["success"]
    future_success = judgments["future_success"]
    reasoning_is_correct = judgments["reasoning_is_correct"]

    valid_domain = (
        (success is not None and success >= success_threshold) and
        (future_success is not None and future_success >= future_success_threshold) and
        (reasoning_is_correct is not None and reasoning_is_correct >= reasoning_is_correct_threshold)
    )

    return valid_domain


def prepare_messages(
    examples: dict,
    base_data_dir: str = "data-v3",
    agent: BrowserAgent = None,
) -> dict:

    observations_dir = os.path.join(
        base_data_dir,
        "observations"
    )

    actions_dir = os.path.join(
        base_data_dir,
        "actions"
    )

    domains = examples["domain"]
    instructions = examples["task"]

    prompts = []
    completions = []
    matched_actions = []
    
    for domain, instruction in zip(domains, instructions):

        observations_path = os.path.join(
            observations_dir,
            "{}.json".format(domain)
        )

        actions_path = os.path.join(
            actions_dir,
            "{}.json".format(domain)
        )

        try:  # file may be corrupted

            with open(observations_path, "r") as file:
                observations = json.load(file)

            with open(actions_path, "r") as file:
                actions = json.load(file)

        except json.decoder.JSONDecodeError:

            continue

        for last_timestep in range(1, len(observations)):

            first_timestep = max(
                0, last_timestep - 
                agent.config.max_history - 1
            )

            obs = observations[first_timestep:last_timestep]
            act = actions[first_timestep:last_timestep]

            current_message = [{
                "role": "system",
                "content": agent.system_prompt
            }]

            ground_truth = None

            for obs_t, act_t in zip(obs, act):

                invalid_step = (
                    obs_t['processed_text'] is None or
                    act_t['response'] is None
                )

                if invalid_step:

                    continue

                user_prompt = agent.get_user_prompt(
                    observation = obs_t['processed_text'],
                    instruction = instruction
                )

                current_message.append({
                    "role": "user",
                    "content": user_prompt
                })

                current_message.append({
                    "role": "assistant",
                    "content": act_t['response']
                })

                ground_truth = act_t['matched_response']

            prompts.append(
                current_message[:-1]
            )

            completions.append(
                current_message[-1]
            )

            matched_actions.append(
                ground_truth
            )

    return {
        "prompt": prompts,
        "completion": completions,
        "ground_truth": matched_actions
    }


if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        type = str,
        default="data-for-agents/insta-150k-v2"
    )

    parser.add_argument(
        "--dataset_split",
        type = str,
        default="train"
    )

    parser.add_argument(
        "--dataset_output_dir",
        type = str,
        default="./insta-150k-v2-grpo"
    )

    parser.add_argument(
        "--base_data_dir",
        type = str,
        default="data-v3"
    )

    parser.add_argument(
        "--max_history",
        type = int,
        default = 2
    )

    parser.add_argument(
        "--max_obs_length",
        type = int,
        default = 2048
    )

    parser.add_argument(
        "--success_threshold",
        type = float,
        default = 0.9
    )

    parser.add_argument(
        "--future_success_threshold",
        type = float,
        default = 0.9
    )

    parser.add_argument(
        "--reasoning_is_correct_threshold",
        type = float,
        default = 0.9
    )

    args = parser.parse_args()

    dataset = load_dataset(
        args.dataset_name,
        split = args.dataset_split
    )

    agent_config = get_agent_config(
        max_history = args.max_history,
        max_obs_tokens = args.max_obs_length,
    )

    agent: BrowserAgent = BrowserAgent(
        agent_config,
        action_parser = "json"
    )

    # client cannot be pickled
    agent.llm_client = None

    select_valid_samples = partial(
        select_valid_samples,
        base_data_dir = args.base_data_dir,
        success_threshold = args.success_threshold,
        future_success_threshold = args.future_success_threshold,
        reasoning_is_correct_threshold = args.reasoning_is_correct_threshold
    )
    
    dataset = dataset.filter(
        select_valid_samples
    )

    prepare_messages = partial(
        prepare_messages,
        base_data_dir = args.base_data_dir,
        agent = agent
    )

    dataset = dataset.map(
        prepare_messages,
        batched = True,
        remove_columns = dataset.column_names,
        batch_size = 32,
        num_proc = 32,
    )
    
    dataset.save_to_disk(
        args.dataset_output_dir
    )