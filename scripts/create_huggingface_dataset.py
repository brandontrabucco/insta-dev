from PIL import Image
from typing import Dict, List, Generator

import datasets
import json
import os

import pyarrow as pa

import argparse
import copy
import tree


GeneratorType = Generator[
    Dict[str, List[Dict[str, any]]],
    None, None
]


DATASET_SCHEMA = datasets.Features({
    "domain": datasets.Value("string"),
    "task": datasets.Value("string"),
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
    }
})


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = 'Create Huggingface dataset'
    )

    parser.add_argument(
        "--base_dataset_dir",
        type = str,
        default = "./data/"
    )

    parser.add_argument(
        "--remove_html",
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

    args = parser.parse_args()

    dataset = datasets.load_dataset(
        args.dataset,
        split = args.dataset_split
    )

    domain_to_task = {
        x["domain"]: x["task"]
        for x in dataset
    }

    base_dataset_dir = args.base_dataset_dir

    observations_dir = os.path.join(
        base_dataset_dir,
        "observations"
    )

    actions_dir = os.path.join(
        base_dataset_dir,
        "actions"
    )

    judgments_dir = os.path.join(
        base_dataset_dir,
        "judgments"
    )

    all_judgment_files = os.listdir(judgments_dir)

    def generate_examples(
        sharded_examples: List[str]
    ) -> GeneratorType:
        
        image_feature = datasets.Image(decode = True, id = None)

        for example_basename in sharded_examples:

            domain = example_basename.removesuffix(".json")
            task = domain_to_task[domain]

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

                observation["metadata"] = list(
                    (observation["metadata"] or {}).values()
                )

                if args.remove_html:

                    observation.pop("raw_html")

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
                "task": task,
                "observations": out_observations,
                "actions": actions,
                "judgment": judgment
            }

            yield output

    dataset = datasets.Dataset.from_generator(
        generate_examples,
        gen_kwargs = {"sharded_examples": all_judgment_files},
        num_proc = args.num_workers,
        features = DATASET_SCHEMA
    )
