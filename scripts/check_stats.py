from insta.utils import (
    VALUE_KEYS
)

import glob
import json
import argparse
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = 'Check stats of agent trajectories'
    )

    parser.add_argument(
        '--actions_dir',
        type = str,
        default = 'data/actions',
        help = 'Directory containing judgment files'
    )

    parser.add_argument(
        '--judgments_dir',
        type = str,
        default = 'data/judgments',
        help = 'Directory containing judgment files'
    )

    args = parser.parse_args()

    judgment_file_pattern = os.path.join(
        args.judgments_dir, '*.json'
    )

    all_judgments = []
    all_actions = []

    for judgment_file in glob.glob(judgment_file_pattern):

        with open(judgment_file, 'r') as file:

            judgment = json.load(file)

        all_judgments.append(judgment)

        actions_file = os.path.join(
            args.actions_dir,
            os.path.basename(judgment_file)
        )

        with open(actions_file, 'r') as file:

            actions = json.load(file)

        all_actions.append(actions)

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

    def get_cdf_score(value_key, threshold):

        cdf_score = sum(
            comparator(judgment.get(value_key) or 0, threshold)
            for judgment in all_judgments
        ) / len(all_judgments)

        return cdf_score

    for threshold in [0.5, 0.7, 0.9, 1.0]:

        cdf_scores = {
            value_key: get_cdf_score(value_key, threshold)
            for value_key in VALUE_KEYS
        }
    
        print('Judge(Success) {} {}: {}\n'.format(
            "==" if threshold == 1 else ">",
            threshold, json.dumps(cdf_scores, indent = 4)
        ))