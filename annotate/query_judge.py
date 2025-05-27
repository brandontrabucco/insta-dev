from insta import (
    get_judge_config,
    JudgeConfig,
    BrowserJudge,
    DEFAULT_JUDGE_CONFIG,
    NULL_JUDGMENT
)

from insta.pipeline import (
    JUDGE_STEPS_TEMPLATE,
    JUDGE_CRITERIA_TEMPLATE
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

from insta.utils import (
    VALUE_KEYS
)


def query_judge(
    example_id: int, dataset: Dataset,
    judge_config: JudgeConfig = DEFAULT_JUDGE_CONFIG,
    input_observations_dir: str = None,
    input_actions_dir: str = None,
    output_judgments_dir: str = None,
    agent_response_key: str = "response",
    add_steps_to_judge: bool = True,
    add_criteria_to_judge: bool = True,
    skip_finished: bool = False,
):

    example_dict = dataset[example_id]

    domain = example_dict.get(
        "website", example_dict.get("domain")
    )

    instruction = example_dict.get(
        "instruction", example_dict.get("task")
    )

    judge_instruction = instruction

    identifier = example_dict.get(
        "identifier", domain
    )

    steps = example_dict.get(
        "steps", []
    )

    criteria = example_dict.get(
        "criteria", []
    )

    if add_steps_to_judge and len(steps) > 0:

        judge_instruction = JUDGE_STEPS_TEMPLATE.format(
            instruction = judge_instruction, steps = "\n".join(
                "{n}. {x}".format(n = idx + 1, x = part)
                for idx, part in enumerate(steps)
            )
        )

    if add_criteria_to_judge and len(criteria) > 0:

        judge_instruction = JUDGE_CRITERIA_TEMPLATE.format(
            instruction = judge_instruction, criteria = "\n".join(
                "{n}. {x}".format(n = idx + 1, x = part)
                for idx, part in enumerate(criteria)
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

    output_judgment_path = os.path.join(
        output_judgments_dir,
        "{}.json".format(identifier)
    )

    valid_example = (
        os.path.exists(input_actions_path)
        and os.path.exists(input_observations_path)
        and not (skip_finished and os.path.exists(output_judgment_path))
    )

    if not valid_example:

        return None

    with open(input_observations_path, "r") as file:
        
        try: observations = json.load(file)

        except: return None

    with open(input_actions_path, "r") as file:
        
        try: actions = json.load(file)

        except: return None
    
    judge = BrowserJudge(
        config = judge_config
    )

    judgment = judge(
        observations = [
            x["processed_text"]
            for x in observations
        ],
        actions = [
            x[agent_response_key]
            for x in actions
        ],
        instruction = judge_instruction,
    )

    invalid_judgment = (
        judgment is None or 
        judgment == NULL_JUDGMENT
    )

    if invalid_judgment:

        return None

    judgment_values = {
        key: judgment.values.get(key)
        for key in VALUE_KEYS
    }

    judgment = {
        **judgment_values,
        "response": judgment.response,
        "matched_response": judgment.matched_response,
    }

    with open(output_judgment_path, "w") as file:
        
        json.dump(
            judgment,
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
        "--judge_model_name",
        type = str,
        default = "Qwen/Qwen3-235B-A22B-fp8-tput",
    )

    parser.add_argument(
        "--judge_api_key",
        type = str,
        default = os.environ.get("TOGETHER_API_KEY")
    )

    parser.add_argument(
        "--judge_llm_endpoint",
        type = str,
        default = "https://api.together.xyz/v1"
    )

    parser.add_argument(
        "--judge_top_p",
        type = float,
        help = "Sampling Top p for LLMs",
        default = 1.0
    )

    parser.add_argument(
        "--judge_top_k",
        type = int,
        help = "Sampling Top k for LLMs",
        default = None
    )

    parser.add_argument(
        "--judge_temperature",
        type = float,
        help = "Sampling temperature for LLMs",
        default = 0.5
    )

    parser.add_argument(
        "--judge_reasoning_effort",
        type = str,
        help = "Set reasoning mode in certain LLMs",
        default = None,
    )

    parser.add_argument(
        "--judge_disable_thinking_chat_template",
        action = "store_true",
        help = "Turns off reasoning mode in certain LLMs"
    )

    parser.add_argument(
        "--add_steps_to_judge",
        action = "store_true",
        help = "Add the steps to the instruction",
        default = False
    )

    parser.add_argument(
        "--add_criteria_to_judge",
        action = "store_true",
        help = "Add the success criteria to the instruction",
        default = False
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
        "--skip_finished",
        action = "store_true",
        help = "Whether to skip existing judgments",
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

    judge_client_type = "openai"

    judge_client_kwargs = {
        "api_key": args.judge_api_key,
        "base_url": args.judge_llm_endpoint
    }

    judge_generation_kwargs = {
        "model": args.judge_model_name,
        "max_tokens": 1024,
        "top_p": args.judge_top_p,
        "temperature": args.judge_temperature,
        "extra_body": {}
    }

    if args.judge_reasoning_effort:

        judge_generation_kwargs.update({
            "reasoning_effort": 
            args.judge_reasoning_effort
        })

    if args.judge_disable_thinking_chat_template:

        if "chat_template_kwargs" not in judge_generation_kwargs["extra_body"]:

            judge_generation_kwargs["extra_body"].update({
                "chat_template_kwargs": {}
            })

        judge_generation_kwargs["extra_body"]["chat_template_kwargs"].update({
            "enable_thinking": False
        })

    if args.judge_top_k is not None:

        judge_generation_kwargs["extra_body"].update({
            "top_k": args.judge_top_k
        })

    judge_config = get_judge_config(
        client_type = judge_client_type,
        client_kwargs = judge_client_kwargs,
        generation_kwargs = judge_generation_kwargs,
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

    output_judgments_dir = os.path.join(
        args.input_data_dir,
        args.judge_name
    )

    dataset = load_dataset(
        args.dataset,
        split = args.dataset_split
    )

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
        output_judgments_dir,
        exist_ok = True
    )

    progress_bar = tqdm.tqdm(
        desc = "Processing",
        dynamic_ncols = True,
        total = len(out_dataset_ids),
    )

    worker_fn = partial(
        query_judge, dataset = dataset,
        judge_config = judge_config,
        input_observations_dir = input_observations_dir,
        input_actions_dir = input_actions_dir,
        output_judgments_dir = output_judgments_dir,
        agent_response_key = args.agent_response_key,
        add_steps_to_judge = args.add_steps_to_judge,
        add_criteria_to_judge = args.add_criteria_to_judge,
        skip_finished = args.skip_finished
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