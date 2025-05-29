from datasets import load_dataset

from insta.configs.judge_config import (
    JudgeConfig,
    get_judge_config,
    DEFAULT_JUDGE_CONFIG
)

from insta.judge import BrowserJudge
from functools import partial

from typing import List, Dict, Any

import argparse
import random
import json
import os


def select_valid_samples(
    example_dict: dict,
    input_data_dir: str = None,
    judge_name: str = "qwen3-235b-judge",
) -> bool:

    observations_dir = os.path.join(
        input_data_dir,
        "observations"
    )

    actions_dir = os.path.join(
        input_data_dir,
        "actions"
    )

    judgments_dir = os.path.join(
        input_data_dir,
        judge_name
    )

    domain = example_dict.get(
        "website", example_dict.get("domain")
    )

    identifier = example_dict.get(
        "identifier", domain
    )
            
    valid_example = (
        os.path.exists(
            os.path.join(
                observations_dir,
                "{}.json".format(identifier)
            )
        ) and os.path.exists(
            os.path.join(
                actions_dir,
                "{}.json".format(identifier)
            )
        ) and os.path.exists(
            os.path.join(
                judgments_dir,
                "{}.json".format(identifier)
            )
        )
    )

    if not valid_example:

        return False
    
    judgment_path = os.path.join(
        judgments_dir,
        "{}.json".format(identifier)
    )     

    with open(judgment_path, "r") as file:

        try: judgment = json.load(file)

        except json.JSONDecodeError: return False

    success = judgment["success"]
    efficiency = judgment["efficiency"]
    self_correction = judgment["self_correction"]

    success_valid = (
        success is not None and 
        isinstance(success, float) and
        (success >= 0 and success <= 1)
    )

    efficiency_valid = (
        efficiency is not None and 
        isinstance(efficiency, float) and
        (efficiency >= 0 and efficiency <= 1)
    )

    self_correction_valid = (
        self_correction is not None and 
        isinstance(self_correction, float) and
        (self_correction >= 0 and self_correction <= 1)
    )

    valid_example = (
        success_valid and 
        efficiency_valid and 
        self_correction_valid
    )

    return valid_example


def get_prompts(
    observations: List[Dict[str, Any]] = None,
    actions: List[Dict[str, Any]] = None,
    instruction: str = None,
    judgment: Dict[str, Any] = None,
    judge: BrowserJudge = None,
    agent_response_key: str = "response",
) -> List[Dict[str, str]]:

    prompts = judge.get_prompts(
        observations = [
            x['processed_text'] 
            for x in observations
        ],
        actions = [
            x[agent_response_key] 
            for x in actions
        ],
        instruction = instruction,
        last_actions = judge.config.last_actions,
        last_obs = judge.config.last_obs,
    )

    prompts.append({
        "role": "assistant",
        "content": judgment["response"]
    })

    return prompts


def unpack_examples(
    identifier: str = None,
    instruction: str = None,
    observations_dir: str = None,
    actions_dir: str = None,
    judgments_dir: str = None,
    judge: BrowserJudge = None,
    agent_response_key: str = "response",
) -> List[List[Dict[str, str]]]:
    
    observations_path = os.path.join(
        observations_dir,
        "{}.json".format(identifier)
    )

    actions_path = os.path.join(
        actions_dir,
        "{}.json".format(identifier)
    )

    judgment_path = os.path.join(
        judgments_dir,
        "{}.json".format(identifier)
    )

    with open(observations_path, "r") as file:
        
        try: observations = json.load(file)

        except json.JSONDecodeError:
            
            return []

    with open(actions_path, "r") as file:

        try: actions = json.load(file)

        except json.JSONDecodeError:
            
            return []

    with open(judgment_path, "r") as file:

        try: judgment = json.load(file)

        except json.JSONDecodeError:
            
            return []

    prompts = get_prompts(
        observations = observations,
        actions = actions,
        instruction = instruction,
        judgment = judgment,
        judge = judge,
        agent_response_key = agent_response_key,
    )

    valid_prompts = all([
        x["content"] is not None
        for x in prompts
    ])

    if not valid_prompts:

        return []

    return [prompts]


def process_dataset(
    examples: dict,
    input_data_dir: str = None,
    judge_name: str = "qwen3-235b-judge",
    judge: BrowserJudge = None,
    agent_response_key: str = "response",
) -> dict:

    observations_dir = os.path.join(
        input_data_dir,
        "observations"
    )

    actions_dir = os.path.join(
        input_data_dir,
        "actions"
    )

    judgments_dir = os.path.join(
        input_data_dir,
        judge_name
    )

    if "identifier" in examples:

        websites = examples["identifier"]

    elif "website" in examples:

        websites = examples["website"]

    elif "domain" in examples:

        websites = examples["domain"]

    if "instruction" in examples:

        instructions = examples["instruction"]

    elif "task" in examples:

        instructions = examples["task"]

    examples = []
    
    for identifier, instruction in zip(
        websites,
        instructions
    ):

        examples.extend(unpack_examples(
            identifier = identifier,
            instruction = instruction,
            observations_dir = observations_dir,
            actions_dir = actions_dir,
            judgments_dir = judgments_dir,
            agent_response_key = agent_response_key,
            judge = judge,
        ))

    output_dict = {
        "messages": examples
    }

    return output_dict


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
        "--input_data_dir",
        type = str,
        default = "/data/matrix/projects/rsalakhugroup/btrabucc/neurips_data_collection/qwen3-1.7b-10000x-0.9s-qwen3-235b-judge"
    )

    parser.add_argument(
        "--dataset_output_dir",
        type = str,
        default="/data/matrix/projects/rsalakhugroup/btrabucc/neurips_sft_judge/{max_num_samples}x-{judge_name}"
    )

    parser.add_argument(
        "--last_actions",
        type = int,
        default = 5
    )

    parser.add_argument(
        "--last_obs",
        type = int,
        default = 5
    )

    parser.add_argument(
        "--max_obs_length",
        type = int,
        default = 2048
    )

    parser.add_argument(
        "--max_num_samples",
        type = int,
        default = 5000
    )

    parser.add_argument(
        "--judge_name",
        type = str,
        default = "qwen3-235b-judge"
    )

    parser.add_argument(
        "--agent_response_key",
        type = str,
        help = "key for response from the agent",
        default = "response",
    )

    args = parser.parse_args()

    args.dataset_output_dir = (
        args.dataset_output_dir.format(
            **vars(args)
        )
    )

    dataset = load_dataset(
        args.dataset_name,
        split = args.dataset_split
    )

    judge_config = get_judge_config(
        last_actions = args.last_actions,
        last_obs = args.last_obs,
        max_obs_tokens = args.max_obs_length,
        judgment_parser = "simplified_json"
    )

    judge: BrowserJudge = BrowserJudge(
        config = judge_config
    )

    # client cannot be pickled
    judge.llm_client = None

    select_valid_samples = partial(
        select_valid_samples,
        input_data_dir = args.input_data_dir,
        judge_name = args.judge_name,
    )
    
    dataset = dataset.filter(
        select_valid_samples,
        num_proc = 32,
    )

    if args.max_num_samples is not None:

        dataset = dataset.select(
            random.Random(0).sample(
                list(range(len(dataset))),
                min(len(dataset), args.max_num_samples)
            )
        )

    process_dataset = partial(
        process_dataset,
        input_data_dir = args.input_data_dir,
        judge_name = args.judge_name,
        agent_response_key = args.agent_response_key,
        judge = judge
    )

    dataset = dataset.map(
        process_dataset,
        batched = True,
        remove_columns = dataset.column_names,
        batch_size = 32,
        num_proc = 32,
    )
    
    dataset.save_to_disk(
        args.dataset_output_dir
    )