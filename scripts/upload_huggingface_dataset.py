from multiprocessing import Pool
from typing import Dict, List, Generator
from PIL import Image

from insta import (
    OBSERVATION_PROCESSORS,
    BaseProcessor,
    BrowserObservation
)

from insta.observation_processors.markdown_processor import (
    FAILED_MESSAGE
)

import datasets

import random
import json
import os

import argparse
import copy
import tree
import tqdm


GeneratorType = Generator[
    Dict[str, List[Dict[str, any]]],
    None, None
]


HIGH_LEVEL_FEATURES = [
    "domain",
    "original_task",
    "observations",
    "actions",
    "judgment",
    "task_proposal"
]


DATASET_SCHEMA = datasets.Features({

    "domain": datasets.Value("string"),
    "original_task": datasets.Value("string"),

    "observations": datasets.Sequence(feature = datasets.Features({

        "current_url": datasets.Value("string"),
        "processed_text": datasets.Value("string"),
        "raw_html": datasets.Value("string"),
        "screenshot": datasets.Image(),

        "metadata": datasets.Sequence(feature = datasets.Features({

            "backend_node_id": datasets.Value("int32"),

            "bounding_client_rect": {
                "x": datasets.Value("float32"),
                "y": datasets.Value("float32"),
                "width": datasets.Value("float32"),
                "height": datasets.Value("float32"),
                "top": datasets.Value("float32"),
                "right": datasets.Value("float32"),
                "bottom": datasets.Value("float32"),
                "left": datasets.Value("float32")
            },

            "computed_style": {
                "display": datasets.Value("string")
            },

            "scroll_left": datasets.Value("float32"),
            "scroll_top": datasets.Value("float32"),

            "editable_value": datasets.Value("string"),

            "is_visible": datasets.Value("bool"),
            "is_frontmost": datasets.Value("bool")

        }))

    })),

    "actions": datasets.Sequence(feature = datasets.Features({

        "function_calls": datasets.Sequence(feature = datasets.Features({
            "dotpath": datasets.Value("string"),
            "args": datasets.Value("string")
        })),

        "response": datasets.Value("string"),
        "matched_response": datasets.Value("string")

    })),

    "judgment": {
        
        "task_is_feasible": datasets.Value("float32"),
        "success": datasets.Value("float32"),
        "on_right_track": datasets.Value("float32"),

        "response": datasets.Value("string"),
        "matched_response": datasets.Value("string")

    },

    "task_proposal": {

        "proposed_task": datasets.Value("string"),
        "task_is_feasible": datasets.Value("float32"),
        "estimated_difficulty": datasets.Value("float32"),
        "estimated_steps": datasets.Value("int32"),

        "response": datasets.Value("string"),
        "matched_response": datasets.Value("string")

    }
})


STR_ENCODING_STRATEGY = ('utf-8', 'replace')


def process_example(
    example_basename: str
) -> Dict[str, List[Dict[str, any]]]:
    
    image_feature = datasets.Image(decode = True, id = None)

    domain = example_basename.removesuffix(".json")
    task = domain_to_task[domain]
    proposal = domain_to_proposal[domain]

    observations_path = os.path.join(
        observations_dir,
        example_basename
    )

    actions_path = os.path.join(
        actions_dir,
        example_basename
    )

    judgment_path = os.path.join(
        judgments_dir,
        example_basename
    )
    
    with open(observations_path, "r") as observations_file:

        observations = json.load(
            observations_file
        )

    with open(actions_path, "r") as actions_file:

        actions = json.load(
            actions_file
        )

    with open(judgment_path, "r") as judgment_file:

        judgment = json.load(
            judgment_file
        )

    out_observations = []

    for observation in observations:

        observation = copy.deepcopy(observation)

        if args.reprocess_observations and (
            observation["raw_html"] is not None and 
            observation["metadata"] is not None
        ):

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

        observation["metadata"] = list(
            (observation["metadata"] or {}).values()
        )

        if not args.keep_html:

            observation.pop(
                "raw_html", None
            )

        observation.pop(
            "screenshot", None
        )

        screenshot_path = observation.pop(
            "screenshot_path", None
        )

        screenshot = None

        if screenshot_path is not None and \
                os.path.exists(screenshot_path):

            screenshot = image_feature.encode_example(
                Image.open(screenshot_path)
                .convert("RGB")
            )

        observation["screenshot"] = screenshot

        out_observations.append(
            observation
        )

    output = {
        "domain": domain,
        "original_task": task,
        "observations": out_observations,
        "actions": actions,
        "judgment": judgment,
        "task_proposal": proposal
    }

    output = tree.map_structure(
        lambda x: x if not isinstance(x, str) else x.encode(
            *STR_ENCODING_STRATEGY
        ).decode(), output
    )

    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = 'Create Huggingface dataset'
    )

    parser.add_argument(
        "--hub_identifier",
        type = str,
        default = "btrabucco/insta-150k-traces-{rank:02d}-of-{world_size:02d}"
    )

    parser.add_argument(
        "--base_dataset_dir",
        type = str,
        default = "./data/"
    )

    parser.add_argument(
        "--task_proposals_file",
        type = str,
        default = "all-tasks.json"
    )

    parser.add_argument(
        "--keep_html",
        action = "store_true"
    )

    parser.add_argument(
        "--num_workers",
        type = int,
        default = 32
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
        "--reprocess_observations",
        action = "store_true",
    )

    parser.add_argument(
        "--seed",
        type = int,
        default = 0
    )

    parser.add_argument(
        "--rank",
        type = int,
        default = 0
    )

    parser.add_argument(
        "--world_size",
        type = int,
        default = 1
    )

    args = parser.parse_args()

    observation_processor: BaseProcessor = (
        OBSERVATION_PROCESSORS[
            args.observation_processor]()
    )

    dataset = datasets.load_dataset(
        args.dataset,
        split = args.dataset_split
    )

    domain_to_task = {
        x["domain"]: x["task"]
        for x in dataset
    }

    observations_dir = os.path.join(
        args.base_dataset_dir,
        "observations"
    )

    actions_dir = os.path.join(
        args.base_dataset_dir,
        "actions"
    )

    judgments_dir = os.path.join(
        args.base_dataset_dir,
        "judgments"
    )

    all_judgment_files = sorted(os.listdir(judgments_dir))

    random.seed(args.seed)
    random.shuffle(all_judgment_files)

    rank_files = all_judgment_files[
        args.rank::
        args.world_size
    ]

    with open(args.task_proposals_file, "r") as tasks_file:

        domain_to_proposal = {
            x.pop("domain"): x
            for x in json.load(tasks_file)
        }

    output_examples = {
        key: []
        for key in HIGH_LEVEL_FEATURES
    }

    progress_bar = tqdm.tqdm(
        total = len(dataset) // args.world_size,
        desc = "Processing examples",
        dynamic_ncols = True
    )

    with Pool(args.num_workers) as pool:

        for example in pool.imap_unordered(
            process_example,
            rank_files
        ):
            
            progress_bar.update(1)

            for key in HIGH_LEVEL_FEATURES:

                output_examples[key].extend(
                    example[key]
                )

    dataset = datasets.Dataset.from_dict(
        output_examples,
        features = DATASET_SCHEMA
    )

    dataset.push_to_hub(
        args.hub_identifier.format(
            rank = args.rank,
            world_size = args.world_size
        )
    )
