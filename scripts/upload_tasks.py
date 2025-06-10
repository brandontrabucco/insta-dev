import datasets
import json
import os

import argparse

from typing import List, Dict, Any
from functools import partial
from collections import defaultdict


def load_tasks(
    examples: Dict[str, List[str]],
    input_tasks_dir: str = None
) -> Dict[str, List[str]]:
    
    outputs = defaultdict(list)
    
    for domain in examples["domain"]:

        target_file = os.path.join(
            input_tasks_dir,
            "{}.json".format(domain)
        )

        valid_tasks_file = (
            os.path.exists(target_file)
        )

        if not valid_tasks_file:
            
            continue

        with open(target_file, "r") as file:

            try: task_dict = json.load(file)

            except json.JSONDecodeError as error:
                
                continue

        output_example = {
            "website": domain,
            "instruction": task_dict["proposed_task"],
            "steps": task_dict["steps"],
            "criteria": task_dict["criteria"],
        }
        
        for key, value in output_example.items():

            outputs[key].append(value)

    return outputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = 'Upload tasks to Huggingface'
    )

    parser.add_argument(
        "--input_data_dir",
        type = str,
        default = "/data/matrix/projects/rsalakhugroup/btrabucc/neurips_data_collection/qwen3-1.7b-10000x-0.9s-qwen3-235b-judge-test"
    )

    parser.add_argument(
        "--task_proposer_name",
        type = str,
        default = "gemini-2.5-flash-task-proposer"
    )

    parser.add_argument(
        "--dataset",
        type = str,
        required = True
    )

    parser.add_argument(
        "--dataset_split",
        type = str,
        default = "test"
    )

    args = parser.parse_args()

    input_tasks_dir = os.path.join(
        args.input_data_dir,
        args.task_proposer_name
    )

    domains = [
        x.replace(".json", "")
        for x in os.listdir(input_tasks_dir)
    ]

    dataset = datasets.Dataset.from_dict({
        "domain": domains
    })

    worker_fn = partial(
        load_tasks,
        input_tasks_dir = input_tasks_dir
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