from insta import VALUE_KEYS
import numpy as np

from datasets import load_dataset

import json
import argparse
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = 'Check stats of agent trajectories'
    )

    parser.add_argument(
        '--input_data_dir',
        type = str,
        help = 'Directory containing trajectory data',
        required = True
    )

    parser.add_argument(
        '--judge_name',
        type = str,
        help = 'Name of the judge',
        required = True
    )

    parser.add_argument(
        '--remove_null',
        action = 'store_true',
        help = 'Remove null judgments and actions'
    )

    args = parser.parse_args()

    judgments_dir = os.path.join(
        args.input_data_dir,
        args.judge_name
    )

    actions_dir = os.path.join(
        args.input_data_dir,
        "actions"
    )

    observations_dir = os.path.join(
        args.input_data_dir,
        "observations"
    )

    dataset = load_dataset(
        "data-for-agents/insta-150k-v3",
        split = "train"
    )

    website_to_task = {
        task_dict['website']: task_dict
        for task_dict in dataset
    }

    all_judgments = []
    all_actions = []
    all_tasks = []

    for judgment_file in os.listdir(judgments_dir):

        identifier = judgment_file.replace(
            ".json", ""
        )

        task = website_to_task[
            identifier
        ]

        judgment_file = os.path.join(
            judgments_dir,
            "{}.json".format(identifier)
        )

        actions_file = os.path.join(
            actions_dir,
            "{}.json".format(identifier)
        )

        observations_file = os.path.join(
            observations_dir,
            "{}.json".format(identifier)
        )

        valid_example = (
            os.path.exists(actions_file) and 
            os.path.exists(observations_file) and
            os.path.exists(judgment_file)
        )

        if args.remove_null and not valid_example:

            if os.path.exists(judgment_file):

                os.remove(judgment_file)

            if os.path.exists(actions_file):

                os.remove(actions_file)

            if os.path.exists(observations_file):

                os.remove(observations_file)

            print("Removing: {}".format(
                identifier
            ))

        elif not args.remove_null \
                and not valid_example:
            
            continue

        with open(judgment_file, 'r') as file:

            judgment = json.load(file)

        with open(actions_file, 'r') as file:

            actions = json.load(file)

        data_collection_error = (
            judgment['response'] is None or
            np.mean([action_dict['response'] is None for action_dict in actions]) > 0
        )

        if args.remove_null and data_collection_error:

            if os.path.exists(judgment_file):

                os.remove(judgment_file)

            if os.path.exists(actions_file):

                os.remove(actions_file)

            if os.path.exists(observations_file):

                os.remove(observations_file)

            print("Removing: {}".format(
                identifier
            ))

        elif not args.remove_null \
                and data_collection_error:
            
            continue

        all_judgments.append(judgment)
        all_actions.append(actions)
        all_tasks.append(task)

    total_num_actions = sum(
        len(actions) 
        for actions in all_actions
    )

    print("Number of actions: {}".format(
        total_num_actions
    ))

    print("Number of trajectories: {}".format(
        len(all_actions)
    ))

    average_num_actions = (
        total_num_actions / 
        len(all_actions)
    )

    print("Average number of actions: {:0.2f}\n".format(
        average_num_actions
    ))

    def comparator(x, threshold):

        return (
            x == threshold 
            if threshold == 1 else 
            x > threshold
        )
    
    indices = filter(
        lambda idx: comparator(all_judgments[idx]['success'], 1.0) and comparator(all_judgments[idx]['efficiency'], 1.0),
        range(len(all_judgments))
    )

    indices = sorted(
        indices,
        key = lambda idx: len(all_tasks[idx]['steps']),
        reverse = True
    )

    for idx in indices[:10]:

        print("\n\n---\n\n")

        print(
            json.dumps(all_judgments[idx], indent = 4)
        )

        print(
            json.dumps(all_tasks[idx], indent = 4)
        )