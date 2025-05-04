from insta import (
    get_judge_config,
    JudgeConfig,
    BrowserJudge
)

from insta.utils import (
    VALUE_KEYS
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


def relabel_judgments(
    example_id: int,
    dataset: Dataset = None,
    input_actions_dir: str = None,
    input_observations_dir: str = None,
    output_judgments_dir: str = None,
    judge_config: JudgeConfig = None,
    agent_response_key: str = None,
    overwrite: bool = False,
):

    example_dict = dataset[example_id]

    domain = example_dict.get(
        "url", example_dict.get("domain")
    )

    instruction = example_dict.get(
        "instruction", example_dict.get("task")
    )

    identifier = example_dict.get(
        "identifier", domain
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
        and (overwrite or not os.path.exists(output_judgment_path))
    )

    if not valid_example:

        return None

    with open(input_observations_path, "r") as file:
        
        observations = json.load(file)

    with open(input_actions_path, "r") as file:
        
        actions = json.load(file)
    
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
        instruction = instruction
    )

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
        "--model_name",
        type = str,
        default = "Qwen/Qwen3-235B-A22B-fp8-tput",
    )

    parser.add_argument(
        "--api_key",
        type = str,
        default = os.environ.get("TOGETHER_API_KEY")
    )

    parser.add_argument(
        "--llm_endpoint",
        type = str,
        default = "https://api.together.xyz/v1"
    )

    parser.add_argument(
        "--input_data_dir",
        type = str,
        default = "/data/matrix/projects/rsalakhugroup/btrabucc/insta-150k-v2-qwen3-235b-together"
    )

    parser.add_argument(
        "--output_judgments_dir",
        type = str,
        default = "judgments"
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
        "--num_agents",
        type = int,
        help = "Number of agents per machine",
        default = 32
    )

    parser.add_argument(
        "--agent_response_key",
        type = str,
        help = "key for response from the agent",
        default = "response",
    )

    parser.add_argument(
        "--disable_reasoning_mode",
        action = "store_true",
        help = "Turns off reasoning mode in certain LLMs"
    )

    parser.add_argument(
        "--overwrite",
        action = "store_true",
        help = "Whether to overwrite existing judgments"
    )

    args = parser.parse_args()

    client_kwargs = {
        "api_key": args.api_key,
        "base_url": args.llm_endpoint
    }

    generation_kwargs = {
        "model": args.model_name,
        "max_tokens": 2048,
        "top_p": 1.0,
        "temperature": 0.5,
    }

    if args.disable_reasoning_mode:

        generation_kwargs.update({
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            },
        })

    judge_config = get_judge_config(
        client_kwargs = client_kwargs,
        generation_kwargs = generation_kwargs
    )

    input_actions_dir = os.path.join(
        args.input_data_dir,
        "actions"
    )

    input_observations_dir = os.path.join(
        args.input_data_dir,
        "observations"
    )

    output_judgments_dir = os.path.join(
        args.input_data_dir,
        args.output_judgments_dir
    )

    input_screenshots_dir = os.path.join(
        args.input_data_dir,
        "screenshots"
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
            args.rank * args.num_agents,
            (args.rank + 1) * args.num_agents):

        out_dataset_ids.extend(dataset_ids[
            agent_rank::args.num_agents * args.world_size
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
        relabel_judgments,
        dataset = dataset,
        judge_config = judge_config,
        input_actions_dir = input_actions_dir,
        input_observations_dir = input_observations_dir,
        output_judgments_dir = output_judgments_dir,
        agent_response_key = args.agent_response_key,
        overwrite = args.overwrite
    )
    
    with Pool(processes = args.num_agents) as pool:

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