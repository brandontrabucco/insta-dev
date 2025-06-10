from insta import (
    get_task_proposer_config,
    TaskProposerConfig,
    BrowserTaskProposer,
    DEFAULT_TASK_PROPOSER_CONFIG,
    NULL_TASK_PROPOSAL,
    AGENT_EXPLORATION_TEMPLATE
)

from multiprocessing import Pool
from functools import partial

from datasets import (
    load_dataset,
    Dataset
)

import argparse
import random

import tqdm
import json
import os


DEFAULT_AGENT_RESPONSE_KEY = "response"
DEFAULT_JUDGE_RESPONSE_KEY = "response"


def query_task_proposer(
    example_id: int, dataset: Dataset,
    task_proposer_config: TaskProposerConfig = 
    DEFAULT_TASK_PROPOSER_CONFIG,
    input_observations_dir: str = None,
    input_actions_dir: str = None,
    input_judgments_dir: str = None,
    output_tasks_dir: str = None,
    agent_response_key: str = DEFAULT_AGENT_RESPONSE_KEY,
    judge_response_key: str = DEFAULT_JUDGE_RESPONSE_KEY,
    skip_finished: bool = False,
) -> str | None:

    example_dict = dataset[example_id]

    website = example_dict.get(
        "website", example_dict.get("domain")
    )

    identifier = example_dict.get(
        "identifier", website
    )

    instruction = example_dict.get(
        "instruction", example_dict.get(
            "task", AGENT_EXPLORATION_TEMPLATE.format(
                website = website
            )
        )
    )

    input_observations_path = os.path.join(
        input_observations_dir,
        "{}.json".format(identifier)
    )

    input_actions_path = os.path.join(
        input_actions_dir,
        "{}.json".format(identifier)
    )

    input_judgment_path = os.path.join(
        input_judgments_dir,
        "{}.json".format(identifier)
    )

    output_task_path = os.path.join(
        output_tasks_dir,
        "{}.json".format(identifier)
    )

    valid_example = (
        os.path.exists(input_actions_path)
        and os.path.exists(input_observations_path)
        and os.path.exists(input_judgment_path)
        and not (skip_finished and os.path.exists(output_task_path))
    )

    if not valid_example:

        return None

    with open(input_observations_path, "r") as file:
        
        try: observations = json.load(file)

        except: return None

    with open(input_actions_path, "r") as file:
        
        try: actions = json.load(file)

        except: return None

    with open(input_judgment_path, "r") as file:

        try: judgment = json.load(file)

        except: return None
    
    task_proposer = BrowserTaskProposer(
        config = task_proposer_config
    )

    task_proposal = task_proposer(
        instruction = instruction,
        website = website,
        observations = [
            x["processed_text"]
            for x in observations
        ],
        actions = [
            x[agent_response_key]
            for x in actions
        ],
        judgment = (
            judgment[
                judge_response_key
            ]
        ),
    )

    invalid_task_proposal = (
        task_proposal is None or 
        task_proposal == NULL_TASK_PROPOSAL
    )

    if invalid_task_proposal:

        return None

    task_proposal = {
        "proposed_task": task_proposal.proposed_task,
        "steps": task_proposal.steps,
        "criteria": task_proposal.criteria,
        "response": task_proposal.response,
        "matched_response": task_proposal.matched_response,
    }

    with open(output_task_path, "w") as file:
        
        json.dump(
            task_proposal,
            file,
            indent = 4
        )

    return identifier


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_data_dir",
        type = str,
        default = "/data/matrix/projects/rsalakhugroup/btrabucc/neurips_data_collection/qwen3-1.7b-10000x-0.9s-qwen3-235b-judge"
    )

    parser.add_argument(
        "--task_proposer_model_name",
        type = str,
        default = "gemini-2.5-flash-preview-04-17",
    )

    parser.add_argument(
        "--task_proposer_api_key",
        type = str,
        default = os.environ.get("GOOGLE_API_KEY"),
    )

    parser.add_argument(
        "--task_proposer_llm_endpoint",
        type = str,
        default = "https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    parser.add_argument(
        "--task_proposer_top_p",
        type = float,
        help = "Sampling Top p for LLMs",
        default = 1.0
    )

    parser.add_argument(
        "--task_proposer_top_k",
        type = int,
        help = "Sampling Top k for LLMs",
        default = None
    )

    parser.add_argument(
        "--task_proposer_temperature",
        type = float,
        help = "Sampling temperature for LLMs",
        default = 0.5
    )

    parser.add_argument(
        "--task_proposer_reasoning_effort",
        type = str,
        help = "Set reasoning mode in certain LLMs",
        default = None,
    )

    parser.add_argument(
        "--task_proposer_disable_thinking_chat_template",
        action = "store_true",
        help = "Turns off reasoning mode in certain LLMs"
    )

    parser.add_argument(
        "--task_proposer_name",
        type = str,
        default = "gemini-2.5-flash-task-proposer"
    )

    parser.add_argument(
        "--judge_name",
        type = str,
        default = "qwen3-235b-judge"
    )

    parser.add_argument(
        "--dataset",
        type = str,
        default = "data-for-agents/insta-150k-v2",
    )

    parser.add_argument(
        "--dataset_split",
        type = str,
        default = "train",
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

    parser.add_argument(
        "--skip_finished",
        action = "store_true",
        help = "Whether to skip existing task proposals",
        default = False
    )

    parser.add_argument(
        "--set_exploration_mode",
        action = "store_true",
        help = "Set the agent to exploration mode",
        default = False
    )

    parser.add_argument(
        "--seed",
        type = int,
        help = "Seed for the dataset",
        default = 0
    )

    parser.add_argument(
        "--rank",
        type = int,
        help = "Rank of the process",
        default = 0
    )

    parser.add_argument(
        "--world_size",
        type = int,
        help = "Number of processes",
        default = 1
    )

    parser.add_argument(
        "--num_workers",
        type = int,
        help = "Number of agents per machine",
        default = 8
    )

    args = parser.parse_args()

    task_proposer_client_type = "openai"

    task_proposer_client_kwargs = {
        "api_key": args.task_proposer_api_key,
        "base_url": args.task_proposer_llm_endpoint
    }

    task_proposer_generation_kwargs = {
        "model": args.task_proposer_model_name,
        "max_tokens": 1024,
        "top_p": args.task_proposer_top_p,
        "temperature": args.task_proposer_temperature,
        "extra_body": {}
    }

    if args.task_proposer_reasoning_effort:

        task_proposer_generation_kwargs.update({
            "reasoning_effort": 
            args.task_proposer_reasoning_effort
        })

    if args.task_proposer_disable_thinking_chat_template:

        task_proposer_generation_kwargs["extra_body"][
            "chat_template_kwargs"
        ] = {"enable_thinking": False}

    if args.task_proposer_top_k is not None:

        task_proposer_generation_kwargs["extra_body"].update({
            "top_k": args.task_proposer_top_k
        })

    task_proposer_config = get_task_proposer_config(
        client_type = task_proposer_client_type,
        client_kwargs = task_proposer_client_kwargs,
        generation_kwargs = task_proposer_generation_kwargs,
        log_errors = True,
    )

    input_observations_dir = os.path.join(
        args.input_data_dir,
        "observations"
    )

    input_actions_dir = os.path.join(
        args.input_data_dir,
        "actions"
    )

    input_judgments_dir = os.path.join(
        args.input_data_dir,
        args.judge_name
    )

    output_tasks_dir = os.path.join(
        args.input_data_dir,
        args.task_proposer_name
    )

    dataset = load_dataset(
        args.dataset,
        split = args.dataset_split
    )

    if args.set_exploration_mode:

        dataset = dataset.remove_columns(list({
            "instruction", "task", "steps", "criteria"
        } & set(dataset.column_names)))

    dataset_ids = list(range(len(dataset)))

    random.seed(args.seed)
    random.shuffle(dataset_ids)

    out_dataset_ids = []

    for agent_rank in range(
            args.rank * args.num_workers,
            (args.rank + 1) * args.num_workers):

        out_dataset_ids.extend(dataset_ids[
            agent_rank::args.num_workers * args.world_size
        ])

    os.makedirs(
        output_tasks_dir,
        exist_ok = True
    )

    progress_bar = tqdm.tqdm(
        desc = "Processing",
        dynamic_ncols = True,
        total = len(out_dataset_ids),
    )

    worker_fn = partial(
        query_task_proposer, dataset = dataset,
        task_proposer_config = task_proposer_config,
        input_observations_dir = input_observations_dir,
        input_actions_dir = input_actions_dir,
        input_judgments_dir = input_judgments_dir,
        output_tasks_dir = output_tasks_dir,
        agent_response_key = args.agent_response_key,
        judge_response_key = args.judge_response_key,
        skip_finished = args.skip_finished,
    )
    
    with Pool(processes = args.num_workers) as pool:

        for identifier in pool.imap_unordered(
            worker_fn,
            out_dataset_ids
        ):
            
            progress_bar.update()

            if identifier is not None:

                progress_bar.set_description(
                    "Processing {}"
                    .format(identifier)
                )