from multiprocessing import Pool

from datasets import (
    load_dataset,
    Dataset
)

import argparse
import random

import numpy as np
import tqdm
import json

import shutil
import glob
import os


DEFAULT_SUCCESS = 0.0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = 'Check stats of agent trajectories'
    )

    parser.add_argument(
        "--output_dataset_dir",
        type = str,
        default = "/data/matrix/projects/rsalakhugroup/btrabucc/insta-150k-v2-qwen-1.5b-best-of-5",
    )

    parser.add_argument(
        "--data_dir_pattern",
        type = str,
        default = "/data/matrix/projects/rsalakhugroup/btrabucc/insta-150k-v2-qwen-1.5b-grpo-n1-x?",
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
        "--num_workers",
        type = int,
        help = "Number of processes",
        default = 32
    )

    parser.add_argument(
        "--success_threshold",
        type = float,
        default = 0.0
    )

    parser.add_argument(
        "--consistency_threshold",
        type = float,
        default = 1.0
    )

    args = parser.parse_args()

    output_observations_dir = os.path.join(
        args.output_dataset_dir,
        "observations"
    )

    output_actions_dir = os.path.join(
        args.output_dataset_dir,
        "actions"
    )

    output_judgments_dir = os.path.join(
        args.output_dataset_dir,
        "judgments"
    )

    output_screenshot_dir = os.path.join(
        args.output_dataset_dir,
        "screenshots"
    )

    os.makedirs(
        args.output_dataset_dir,
        exist_ok = True
    )

    os.makedirs(
        output_observations_dir,
        exist_ok = True
    )

    os.makedirs(
        output_actions_dir,
        exist_ok = True
    )

    os.makedirs(
        output_judgments_dir,
        exist_ok = True
    )

    os.makedirs(
        output_screenshot_dir,
        exist_ok = True
    )

    dataset = load_dataset(
        args.dataset,
        split = args.dataset_split
    )

    dataset_ids = list(range(len(dataset)))

    random.seed(args.seed)
    random.shuffle(dataset_ids)

    all_data_dirs = list(glob.glob(
        args.data_dir_pattern
    ))

    def worker_function(domain: str) -> float:

        dirs_judgments = [] 

        domain_success_values = []

        for data_dir in all_data_dirs:

            judgment_file = os.path.join(
                data_dir,
                "judgments",
                "{}.json".format(domain)
            )

            if not os.path.exists(
                judgment_file
            ):

                continue

            with open(judgment_file, 'r') as file:

                judgment_dict = json.load(
                    file
                )

            success = judgment_dict.get(
                'success', DEFAULT_SUCCESS
            ) or DEFAULT_SUCCESS

            judgment_dict['success'] = success

            domain_success_values.append(
                success > 0.5
            )

            did_not_succeed = (
                success < 
                args.success_threshold
            )

            if did_not_succeed:

                continue

            dirs_judgments.append(
                (data_dir, judgment_dict)
            )

        if len(dirs_judgments) == 0:

            return 0.0, 0.0

        consistency = np.mean(
            domain_success_values
        )

        already_too_consistent = (
            consistency > 
            args.consistency_threshold
        )

        if already_too_consistent:

            return 0.0, 0.0

        best_data_dir, best_judgment = max(
            dirs_judgments,
            key = lambda x: x[1]['success']
        )

        input_observations_path = os.path.join(
            best_data_dir,
            "observations",
            "{}.json".format(domain)
        )

        input_actions_path = os.path.join(
            best_data_dir,
            "actions",
            "{}.json".format(domain)
        )

        input_judgment_path = os.path.join(
            best_data_dir,
            "judgments",
            "{}.json".format(domain)
        )

        input_screenshot_path = os.path.join(
            best_data_dir,
            "screenshots",
            "{}".format(domain)
        )

        output_observations_path = os.path.join(
            output_observations_dir,
            "{}.json".format(domain)
        )

        output_actions_path = os.path.join(
            output_actions_dir,
            "{}.json".format(domain)
        )

        output_judgment_path = os.path.join(
            output_judgments_dir,
            "{}.json".format(domain)
        )

        output_screenshot_path = os.path.join(
            output_screenshot_dir,
            "{}".format(domain)
        )

        shutil.copy(
            input_observations_path,
            output_observations_path
        )

        shutil.copy(
            input_actions_path,
            output_actions_path
        )

        shutil.copy(
            input_judgment_path,
            output_judgment_path
        )

        shutil.copytree(
            input_screenshot_path,
            output_screenshot_path,
            dirs_exist_ok = True
        )

        return best_judgment['success'], consistency

    domains = [
        dataset[idx]['domain']
        for idx in dataset_ids 
    ]

    with Pool(processes = args.num_workers) as pool:

        progress_bar = tqdm.tqdm(
            total = len(domains),
            desc = "Merging trajectories"
        )

        successes = []
        consistencies = []

        for success, consistency in pool.imap_unordered(
            worker_function, 
            domains
        ):

            progress_bar.update()

            successes.append(success > 0.5)
            consistencies.append(consistency)

            mean_success_rate = np.mean(successes)
            mean_consistency_rate = np.mean(consistencies)

            progress_bar.set_description(
                'Success = {:0.3f}, Consistency = {:0.3f}'.format(
                    mean_success_rate,
                    mean_consistency_rate
                )
            )

