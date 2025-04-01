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

        with open(judgment_file, 'r') as f:
            judgment = json.load(f)

        all_judgments.append(judgment)

        actions_file = os.path.join(
            args.actions_dir,
            os.path.basename(judgment_file)
        )

        with open(actions_file, 'r') as f:
            actions = json.load(f)

        all_actions.append(actions)

    total_num_actions = sum(
        len(actions) 
        for actions in all_actions
    )

    average_num_actions = (
        total_num_actions / 
        len(all_actions)
    )

    average_values = {
        key: sum(
            judgment[key] or 0.0 
            for judgment in all_judgments
        ) / len(all_judgments)
        for key in VALUE_KEYS
    }

    fraction_eq_1 = {
        key: sum(
            judgment[key] == 1 
            for judgment in all_judgments
        ) / len(all_judgments)
        for key in VALUE_KEYS
    }

    fraction_ge_0_5 = {
        key: sum(
            (judgment[key] or 0) > 0.5
            for judgment in all_judgments
        ) / len(all_judgments)
        for key in VALUE_KEYS
    }

    print("Number of actions: {}".format(
        total_num_actions
    ))

    print("Number of trajectories: {}".format(
        len(all_actions)
    ))

    print("Average number of actions: {:0.2f}\n".format(
        average_num_actions
    ))

    print('Average: {}\n'.format(
        json.dumps(average_values, indent = 4)
    ))

    print('Fraction conf = 1: {}\n'.format(
        json.dumps(fraction_eq_1, indent = 4)
    ))

    print('Fraction conf > 0.5: {}\n'.format(
        json.dumps(fraction_ge_0_5, indent = 4)
    ))
