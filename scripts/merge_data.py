import os
import shutil
import argparse
import random
import tqdm

from datasets import load_dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_data_dir",
        type = str,
        default = "old-data"
    )

    parser.add_argument(
        "--output_data_dir",
        type = str,
        default = "data"
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

    args = parser.parse_args()

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
        "judgments"
    )

    input_screenshots_dir = os.path.join(
        args.input_data_dir,
        "screenshots"
    )

    output_actions_dir = os.path.join(
        args.output_data_dir,
        "actions"
    )

    output_observations_dir = os.path.join(
        args.output_data_dir,
        "observations"
    )

    output_judgments_dir = os.path.join(
        args.output_data_dir,
        "judgments"
    )

    output_screenshots_dir = os.path.join(
        args.output_data_dir,
        "screenshots"
    )

    dataset = load_dataset(
        args.dataset,
        split = args.dataset_split
    )

    dataset_ids = list(range(len(dataset)))

    random.seed(args.seed)
    random.shuffle(dataset_ids)

    dataset_ids = dataset_ids[
        args.rank::args.world_size
    ]

    progress_bar = tqdm.tqdm(
        dataset_ids, desc = "Processing",
        dynamic_ncols = True
    )

    for example_id in progress_bar:

        example_dict = dataset[example_id]

        domain = example_dict["domain"]

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

        input_screenshots_path = os.path.join(
            input_screenshots_dir,
            "{}".format(domain)
        )

        output_actions_path = os.path.join(
            output_actions_dir,
            "{}.json".format(domain)
        )

        output_observations_path = os.path.join(
            output_observations_dir,
            "{}.json".format(domain)
        )

        output_judgment_path = os.path.join(
            output_judgments_dir,
            "{}.json".format(domain)
        )

        output_screenshots_path = os.path.join(
            output_screenshots_dir,
            "{}".format(domain)
        )

        # check if the target path exists in output
        # if not copy from input to output

        if not os.path.exists(output_actions_path) \
                and os.path.exists(input_actions_path):

            shutil.copy(
                input_actions_path,
                output_actions_path
            )

        if not os.path.exists(output_observations_path) \
                and os.path.exists(input_observations_path):

            shutil.copy(
                input_observations_path,
                output_observations_path
            )

        if not os.path.exists(output_judgment_path) \
                and os.path.exists(input_judgment_path):

            shutil.copy(
                input_judgment_path,
                output_judgment_path
            )

        if not os.path.exists(output_screenshots_path) \
                and os.path.exists(input_screenshots_path):

            shutil.copytree(
                input_screenshots_path,
                output_screenshots_path
            )