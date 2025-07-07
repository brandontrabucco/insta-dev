from collections import defaultdict
from contextlib import suppress

from typing import List, Dict
from functools import partial

import datasets
import json
import os
import argparse


def comparator(x: float, threshold: float) -> bool:

    return (
        x == threshold 
        if threshold == 1 else 
        x >  threshold
    )


def load_tasks(
    examples: Dict[str, List[str]],
    judgments_dir: str,
    task_proposals_dir: str,
    success_threshold: float | None = None,
) -> Dict[str, List[str]]:
    
    outputs = defaultdict(list)
    
    for domain in examples["domain"]:

        task_proposal_dict = None
        judgment_dict = None

        task_proposal_path = os.path.join(
            task_proposals_dir,
            "{}.json".format(domain)
        )

        judgment_path = os.path.join(
            judgments_dir,
            "{}.json".format(domain)
        )

        files_exist = (
            os.path.exists(task_proposal_path) and 
            os.path.exists(judgment_path)
        )

        if not files_exist:
            continue

        with open(task_proposal_path, "r") as task_proposal_file:
            with suppress(json.JSONDecodeError):
                task_proposal_dict = json.load(task_proposal_file)

        if task_proposal_dict is None:
            continue

        with open(judgment_path, "r") as judgment_file:
            with suppress(json.JSONDecodeError):
                judgment_dict = json.load(judgment_file)

        if judgment_dict is None:
            continue

        has_missing_keys = (
            task_proposal_dict["proposed_task"] is None or
            task_proposal_dict["steps"] is None or
            task_proposal_dict["criteria"] is None or 
            judgment_dict["success"] is None
        )

        if has_missing_keys:
            continue

        is_already_solved = (
            success_threshold is not None and 
            comparator(judgment_dict["success"], success_threshold)
        )

        if is_already_solved:
            continue

        instruction = task_proposal_dict["proposed_task"]
        steps = task_proposal_dict["steps"]
        criteria = task_proposal_dict["criteria"]

        output_example = {
            "website": domain,
            "instruction": instruction,
            "steps": steps,
            "criteria": criteria,
        }
        
        for k, v in output_example.items():
            outputs[k].append(v)

    return outputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = 'Upload tasks to Huggingface'
    )

    parser.add_argument(
        "--input_data_dir",
        type = str,
        default = "./"
    )

    parser.add_argument(
        "--judge_name",
        type = str,
        default = "gemini-2.5-flash-judge"
    )

    parser.add_argument(
        "--task_proposer_name",
        type = str,
        default = "gemini-2.5-flash-task-refiner"
    )

    parser.add_argument(
        "--dataset",
        type = str,
        required = True
    )

    parser.add_argument(
        "--dataset_split",
        type = str,
        default = "train"
    )

    parser.add_argument(
        "--success_threshold",
        type = float,
        default = None
    )

    args = parser.parse_args()

    task_proposals_dir = os.path.join(
        args.input_data_dir,
        args.task_proposer_name
    )

    judgments_dir = os.path.join(
        args.input_data_dir,
        args.judge_name
    )

    domains = [
        x.replace(".json", "")
        for x in os.listdir(task_proposals_dir)
    ]

    dataset = datasets.Dataset.from_dict({
        "domain": domains
    })

    worker_fn = partial(
        load_tasks,
        judgments_dir = judgments_dir,
        task_proposals_dir = task_proposals_dir,
        success_threshold = args.success_threshold
    )

    dataset = dataset.map(
        worker_fn,
        batched = True,
        remove_columns = dataset.column_names,
        load_from_cache_file = False,
        batch_size = 32,
        num_proc = 32,
    )

    dataset.push_to_hub(
        args.dataset,
        split = args.dataset_split
    )