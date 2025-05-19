from insta import (
    get_task_proposer_config,
    TaskProposerConfig,
    BrowserTaskProposer,
    DEFAULT_TASK_PROPOSER_CONFIG
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


def get_task_proposals(
    example_id: int,
    dataset: Dataset = None,
    input_actions_dir: str = None,
    input_observations_dir: str = None,
    input_judgments_dir: str = None,
    task_proposer_config: TaskProposerConfig = 
    DEFAULT_TASK_PROPOSER_CONFIG
):
    
    task_proposer = BrowserTaskProposer(
        config = task_proposer_config
    )

    example_dict = dataset[example_id]

    domain = example_dict["domain"]
    task = example_dict["task"]

    input_actions_path = os.path.join(
        input_actions_dir,
        "{}.json".format(domain)
    )

    input_observations_path = os.path.join(
        input_observations_dir,
        "{}.json".format(domain)
    )

    input_judgment_path = os.path.join(
        input_judgments_dir,
        "{}.json".format(domain)
    )

    valid_example = (
        os.path.exists(input_actions_path)
        and os.path.exists(input_observations_path)
        and os.path.exists(input_judgment_path)
    )

    if not valid_example:

        return None, None

    with open(input_actions_path, "r") as file:
        
        actions = json.load(
            file
        )

    with open(input_observations_path, "r") as file:
        
        observations = json.load(
            file
        )

    with open(input_judgment_path, "r") as file:
        
        judgment = json.load(
            file
        )

    task_proposal = task_proposer(
        observations = [
            x.get("processed_text")
            for x in observations
        ],
        actions = [
            x.get("response")
            for x in actions
        ],
        judgment = (
            judgment.get("response")
        ),
        instruction = (
            task
        ),
        target_url = domain
    )

    task_proposal = {
        "proposed_task": task_proposal.proposed_task,
        "steps": task_proposal.steps,
        "criteria": task_proposal.criteria,
        "response": task_proposal.response,
        "matched_response": task_proposal.matched_response,
    }

    return domain, task_proposal


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type = str,
        default = "gemini-2.5-flash-preview-04-17",
    )

    parser.add_argument(
        "--api_key",
        type = str,
        default = os.environ.get("GOOGLE_API_KEY"),
    )

    parser.add_argument(
        "--llm_endpoint",
        type = str,
        default = "https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    parser.add_argument(
        "--reasoning_effort",
        type = str,
        help = "Set reasoning mode in certain LLMs",
        default = None,
    )

    parser.add_argument(
        "--disable_thinking_chat_template",
        action = "store_true",
        help = "Turns off reasoning mode in certain LLMs"
    )

    parser.add_argument(
        "--input_data_dir",
        type = str,
        default = os.path.join(
            os.environ.get("NFS_DIR"),
            "btrabucc/neurips_data_collection",
            "qwen3-1.7b-10000x-0.9s-qwen3-235b-judge"
        )
    )

    parser.add_argument(
        "--judge_name",
        type = str,
        default = "gpt-4.1-nano-judge"
    )

    parser.add_argument(
        "--output_tasks_file",
        type = str,
        default = "insta-150k-v3.json"
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
        "max_tokens": 1024,
        "top_p": 1.0,
        "temperature": 0.5,
        "extra_body": {}
    }

    if args.reasoning_effort:

        generation_kwargs.update({
            "reasoning_effort": 
            args.reasoning_effort
        })

    if args.disable_thinking_chat_template:

        generation_kwargs["extra_body"][
            "chat_template_kwargs"
        ] = {"enable_thinking": False}

    task_proposer_config = get_task_proposer_config(
        client_kwargs = client_kwargs,
        generation_kwargs = generation_kwargs,
        log_errors = True
    )

    input_actions_dir = os.path.join(
        args.input_data_dir,
        "actions"
    )

    input_observations_dir = os.path.join(
        args.input_data_dir,
        "observations"
    )

    input_judgments_dir = os.path.join(
        args.input_data_dir,
        args.judge_name
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

    all_tasks = []

    if not args.overwrite and os.path.exists(args.output_tasks_file):

        with open(args.output_tasks_file, "r") as file:

            all_tasks = json.load(file)

        finished_domains = set([
            example_dict['domain']
            for example_dict in all_tasks
        ])

        dataset_ids = [
            example_idx for example_idx in dataset_ids
            if dataset[example_idx]["domain"]
            not in finished_domains
        ]

    out_dataset_ids = []

    for agent_rank in range(
            args.rank * args.num_agents,
            (args.rank + 1) * args.num_agents):

        out_dataset_ids.extend(dataset_ids[
            agent_rank::args.num_agents * args.world_size
        ])

    progress_bar = tqdm.tqdm(
        desc = "Processing",
        dynamic_ncols = True,
        total = len(out_dataset_ids),
    )

    worker_fn = partial(
        get_task_proposals,
        dataset = dataset,
        input_actions_dir = input_actions_dir,
        input_observations_dir = input_observations_dir,
        input_judgments_dir = input_judgments_dir,
        task_proposer_config = task_proposer_config
    )
    
    with Pool(processes = args.num_agents) as pool:

        for domain, task_dict in pool.imap_unordered(
            worker_fn,
            out_dataset_ids
        ):
            
            progress_bar.update()

            if domain is not None and task_dict is not None:

                progress_bar.set_description(
                    "Processing {}"
                    .format(domain)
                )

                all_tasks.append({
                    "domain": domain,
                    **task_dict
                })
                
                if len(all_tasks) % 100 == 0:

                    with open(args.output_tasks_file, "w") as file:

                        json.dump(
                            all_tasks,
                            file,
                            indent = 4
                        )

    with open(args.output_tasks_file, "w") as file:

        json.dump(
            all_tasks,
            file,
            indent = 4
        )