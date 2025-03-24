from insta import (
    OBSERVATION_PROCESSORS,
    BaseProcessor,
    BrowserObservation,
)

from insta.observation_processors.markdown_processor import (
    FAILED_MESSAGE
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

import copy
import json
import os


def process_observations(
    example_id: int,
    dataset: Dataset = None,
    observation_processor: BaseProcessor = None,
    args: argparse.Namespace = None
):
    
    domain = dataset[example_id]["domain"]

    observations_path = os.path.join(
        args.input_observations_dir,
        "{}.json".format(domain)
    )

    with open(observations_path, "r") as observations_file:

        observations = json.load(
            observations_file
        )

    out_observations = []

    for observation in observations:

        observation = copy.deepcopy(observation)

        observation_is_valid = (
            observation["metadata"] is not None and
            observation["raw_html"] is not None
        )

        if observation_is_valid:

            updated_metadata = copy.deepcopy(
                observation["metadata"]
            )

            browser_obs = BrowserObservation(
                raw_html = observation["raw_html"],
                metadata = updated_metadata,
                current_url = observation["current_url"]
            )

            updated_obs = observation_processor.process(
                observation = browser_obs,
                restrict_viewport = args.restrict_viewport,
                require_visible = not args.not_require_visible,
                require_frontmost = not args.not_require_frontmost,
                remove_pii = args.remove_pii
            )

            processing_failed = (
                updated_obs.processed_text ==
                FAILED_MESSAGE
            )

            if not processing_failed:

                observation["processed_text"] = (
                    updated_obs.processed_text
                )

        out_observations.append(
            observation
        )

    output_observations_path = os.path.join(
        args.output_observations_dir,
        "{}.json".format(domain)
    )

    with open(output_observations_path, "w") as output_observations_file:

        json.dump(
            out_observations,
            output_observations_file,
            indent = 4
        )

    return domain


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
        "--observation_processor",
        type = str,
        default = "markdown",
    )

    parser.add_argument(
        "--restrict_viewport",
        type = int, nargs = "+",
        default = (0, 0, 1920, 1080),
    )

    parser.add_argument(
        "--not_require_visible",
        action = "store_true",
    )

    parser.add_argument(
        "--not_require_frontmost",
        action = "store_true",
    )

    parser.add_argument(
        "--remove_pii",
        action = "store_true",
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

    observation_processor: BaseProcessor = (
        OBSERVATION_PROCESSORS[
            args.observation_processor]()
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
        process_observations,
        dataset = dataset,
        observation_processor = observation_processor,
        args = args
    )
    
    with Pool(processes = args.num_agents) as pool:

        for domain in pool.imap_unordered(
            worker_fn,
            out_dataset_ids
        ):
            
            progress_bar.update()

            if domain is not None:

                progress_bar.set_description(
                    "Processing {}"
                    .format(domain)
                )
