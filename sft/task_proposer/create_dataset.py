from datasets import load_dataset

from insta.configs.task_proposer_config import (
    TaskProposerConfig,
    get_task_proposer_config,
    DEFAULT_TASK_PROPOSER_CONFIG
)

from insta.task_proposer import BrowserTaskProposer
from functools import partial

from typing import List, Dict, Any

import argparse
import random
import json
import os


def select_valid_samples(
    example_dict: dict,
    input_data_dir: str = None,
    task_len_threshold: int = 100,
    steps_threshold: int = 3,
    criteria_threshold: int = 3,
    task_proposer_name: str = "gemini-2.5-flash-task-proposer",
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

    tasks_dir = os.path.join(
        input_data_dir,
        task_proposer_name
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
        ) and os.path.exists(
            os.path.join(
                tasks_dir,
                "{}.json".format(identifier)
            )
        )
    )

    if not valid_example:

        return False
    
    task_path = os.path.join(
        tasks_dir,
        "{}.json".format(domain)
    )     

    with open(task_path, "r") as file:

        try: task_dict = json.load(file)
        except json.JSONDecodeError: return False

    proposed_task = task_dict["proposed_task"]
    steps = task_dict["steps"]
    criteria = task_dict["criteria"]

    proposed_task_valid = (
        proposed_task is not None and 
        (task_len_threshold == 0 or len(proposed_task) > task_len_threshold)
    )

    steps_valid = (
        steps is not None and 
        (steps_threshold == 0 or len(steps) > steps_threshold)
    )

    criteria_valid = (
        criteria is not None and 
        (criteria_threshold == 0 or len(criteria) > criteria_threshold)
    )

    valid_example = (
        proposed_task_valid and 
        steps_valid and 
        criteria_valid
    )

    return valid_example


def get_prompts(
    website: str = None,
    previous_task: str = None,
    observations: List[Dict[str, Any]] = None,
    actions: List[Dict[str, Any]] = None,
    judgment: Dict[str, Any] = None,
    proposed_task: Dict[str, Any] = None,
    task_proposer: BrowserTaskProposer = None,
    agent_response_key: str = "response",
    judge_response_key: str = "response",
) -> List[Dict[str, str]]:

    prompts = task_proposer.get_prompts(
        website = website,
        instructions = [previous_task],
        observations = [[
            x['processed_text'] 
            for x in observations
        ]],
        actions = [[
            x[agent_response_key] 
            for x in actions
        ]],
        judgments = [
            judgment[judge_response_key]
        ],
        task_proposals = [],
        last_judgments = task_proposer.config.last_judgments,
        last_tasks = task_proposer.config.last_tasks,
        last_trajectories = task_proposer.config.last_trajectories,
        last_actions = task_proposer.config.last_actions,
        last_obs = task_proposer.config.last_obs,
    )

    prompts.append({
        "role": "assistant",
        "content": proposed_task["response"]
    })

    return prompts


def unpack_examples(
    website: str = None,
    previous_task: str = None,
    observations_dir: str = None,
    actions_dir: str = None,
    judgments_dir: str = None,
    tasks_dir: str = None,
    task_proposer: BrowserTaskProposer = None,
    agent_response_key: str = "response",
    judge_response_key: str = "response",
) -> List[List[Dict[str, str]]]:
    
    observations_path = os.path.join(
        observations_dir,
        "{}.json".format(website)
    )

    actions_path = os.path.join(
        actions_dir,
        "{}.json".format(website)
    )

    judgment_path = os.path.join(
        judgments_dir,
        "{}.json".format(website)
    )

    task_path = os.path.join(
        tasks_dir,
        "{}.json".format(website)
    )

    with open(observations_path, "r") as file:
        
        try: observations = json.load(file)

        except json.JSONDecodeError: return []

    with open(actions_path, "r") as file:

        try: actions = json.load(file)

        except json.JSONDecodeError: return []

    with open(judgment_path, "r") as file:

        try: judgment = json.load(file)

        except json.JSONDecodeError: return []

    with open(task_path, "r") as file:

        try: proposed_task = json.load(file)
        
        except json.JSONDecodeError: return []

    prompts = get_prompts(
        website = website,
        previous_task = previous_task,
        observations = observations,
        actions = actions,
        judgment = judgment,
        proposed_task = proposed_task,
        task_proposer = task_proposer,
        agent_response_key = agent_response_key,
        judge_response_key = judge_response_key,
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
    task_proposer_name: str = "gemini-2.5-flash-task-proposer",
    judge_name: str = "qwen3-235b-judge",
    task_proposer: BrowserTaskProposer = None,
    agent_response_key: str = "response",
    judge_response_key: str = "response",
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

    tasks_dir = os.path.join(
        input_data_dir,
        task_proposer_name
    )

    websites = examples["domain"]
    previous_tasks = examples["task"]

    examples = []
    
    for website, previous_task in zip(
        websites,
        previous_tasks
    ):

        examples.extend(unpack_examples(
            website = website,
            previous_task = previous_task,
            observations_dir = observations_dir,
            actions_dir = actions_dir,
            judgments_dir = judgments_dir,
            tasks_dir = tasks_dir,
            task_proposer = task_proposer,
            agent_response_key = agent_response_key,
            judge_response_key = judge_response_key
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
        default="/data/matrix/projects/rsalakhugroup/btrabucc/neurips_sft_task_proposer/{max_num_samples}x-{task_proposer_name}-{judge_name}"
    )

    parser.add_argument(
        "--last_judgments",
        type = int,
        default = 5
    )

    parser.add_argument(
        "--last_tasks",
        type = int,
        default = 5
    )

    parser.add_argument(
        "--last_trajectories",
        type = int,
        default = 1
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
        "--task_len_threshold",
        type = int,
        default = 100
    )

    parser.add_argument(
        "--steps_threshold",
        type = int,
        default = 3
    )

    parser.add_argument(
        "--criteria_threshold",
        type = int,
        default = 3
    )

    parser.add_argument(
        "--judge_name",
        type = str,
        default = "qwen3-235b-judge"
    )

    parser.add_argument(
        "--task_proposer_name",
        type = str,
        default = "gemini-2.5-flash-task-proposer"
    )

    parser.add_argument(
        "--agent_response_key",
        type = str,
        help = "key for response from the agent",
        default = "response",
    )

    parser.add_argument(
        "--judge_response_key",
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

    task_proposer_config = get_task_proposer_config(
        last_judgments = args.last_judgments,
        last_tasks = args.last_tasks,
        last_trajectories = args.last_trajectories,
        last_actions = args.last_actions,
        last_obs = args.last_obs,
        max_obs_tokens = args.max_obs_length,
        task_parser = "simplified_json"
    )

    task_proposer: BrowserTaskProposer = BrowserTaskProposer(
        config = task_proposer_config
    )

    # client cannot be pickled
    task_proposer.llm_client = None

    select_valid_samples = partial(
        select_valid_samples,
        input_data_dir = args.input_data_dir,
        task_len_threshold = args.task_len_threshold,
        steps_threshold = args.steps_threshold,
        criteria_threshold = args.criteria_threshold,
        task_proposer_name = args.task_proposer_name,
        judge_name = args.judge_name,
    )
    
    dataset = dataset.filter(
        select_valid_samples
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
        task_proposer_name = args.task_proposer_name,
        judge_name = args.judge_name,
        agent_response_key = args.agent_response_key,
        judge_response_key = args.judge_response_key,
        task_proposer = task_proposer
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