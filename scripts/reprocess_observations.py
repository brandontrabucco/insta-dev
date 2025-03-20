from insta import (
    OBSERVATION_PROCESSORS,
    BrowserObservation
)

from insta.observation_processors.markdown_processor import FAILED_MESSAGE

from typing import Tuple
from multiprocessing import Pool
from functools import partial

from datasets import (
    load_dataset,
    Dataset
)

import argparse
import tqdm
import copy

import random
import json
import os


def relabel_observations(
    example_id: str,
    dataset: Dataset = None,
    input_observations_dir: str = None,
    output_observations_dir: str = None,
    observation_processor: str = "markdown",
    restrict_viewport: Tuple[float, float, float, float] = None,
    require_visible: bool = True,
    require_frontmost: bool = True,
    remove_pii: bool = False
) -> str | None:
    
    domain = dataset[example_id]["domain"]

    num_failed = 0

    input_observations_path = os.path.join(
        input_observations_dir,
        "{}.json".format(domain)
    )

    valid_example = os.path.exists(
        input_observations_path
    )

    if not valid_example:

        return None, num_failed

    with open(input_observations_path, "r") as file:
        
        observations = json.load(file)
    
    processor = OBSERVATION_PROCESSORS[
        observation_processor
    ]()

    for original_obs in observations:

        if original_obs["metadata"] is None:

            continue

        updated_metadata = copy.deepcopy(
            original_obs["metadata"]
        )

        for metadata in updated_metadata.values():

            metadata["candidate_id"] = metadata[
                "backend_node_id"
            ]

        obs = BrowserObservation(
            raw_html = original_obs["raw_html"],
            metadata = updated_metadata,
            current_url = original_obs["current_url"]
        )

        updated_obs = processor.process(
            observation = obs,
            restrict_viewport = restrict_viewport,
            require_visible = require_visible,
            require_frontmost = require_frontmost,
            remove_pii = remove_pii
        )

        original_obs["processed_text"] = (
            updated_obs.processed_text
        )

        processing_failed = (
            updated_obs.processed_text ==
            FAILED_MESSAGE
        )

        if processing_failed:

            num_failed += 1

    output_observations_path = os.path.join(
        output_observations_dir,
        "{}.json".format(domain)
    )

    with open(output_observations_path, "w") as file:

        json.dump(
            observations,
            file,
            indent = 4
        )

    return domain, num_failed


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_observations_dir",
        type = str,
        default = "data/observations"
    )

    parser.add_argument(
        "--output_observations_dir",
        type = str,
        default = "data/observations-relabeled"
    )

    parser.add_argument(
        "--observation_processor",
        type = str,
        default = "markdown"
    )

    parser.add_argument(
        "--dataset",
        type = str,
        default = "data-for-agents/insta-150k",
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

    args = parser.parse_args()

    os.makedirs(
        args.output_observations_dir,
        exist_ok = True
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

    progress_bar = tqdm.tqdm(
        desc = "Processing",
        dynamic_ncols = True,
        total = len(out_dataset_ids),
    )

    worker_fn = partial(
        relabel_observations,
        dataset = dataset,
        input_observations_dir = args.input_observations_dir,
        output_observations_dir = args.output_observations_dir,
        observation_processor = args.observation_processor,
        restrict_viewport = (0, 0, 1920, 1080),
        require_visible = True,
        require_frontmost = True,
        remove_pii = False
    )

    num_failed = 0
    
    with Pool(processes = args.num_agents * 2) as pool:

        for domain, failed in pool.imap_unordered(
            worker_fn,
            out_dataset_ids
        ):
            
            num_failed += failed
            
            progress_bar.update()

            if domain is not None:

                progress_bar.set_description(
                    "Processing {}"
                    .format(domain)
                )

            progress_bar.set_postfix(
                failed = num_failed
            )