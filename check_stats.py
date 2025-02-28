import glob
import json
import argparse

import os


VALUES_KEYS = [
    'task_is_feasible',
    'success',
    'on_right_track',
]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = 'Check stats of agent trajectories'
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

    for judgment_file in glob.glob(judgment_file_pattern):

        with open(judgment_file, 'r') as f:
            judgment = json.load(f)

        all_judgments.append(judgment)

    average_values = {
        key: sum(judgment[key] or 0.0 for judgment in all_judgments) / len(all_judgments)
        for key in VALUES_KEYS
    }

    fraction_eq_1 = {
        key: sum(judgment[key] == 1 for judgment in all_judgments) / len(all_judgments)
        for key in VALUES_KEYS
    }

    print("Number of judgments: {}\n".format(
        len(all_judgments)
    ))

    print('Average: {}\n'.format(
        json.dumps(average_values, indent = 4)
    ))

    print('Fration conf = 1: {}'.format(
        json.dumps(fraction_eq_1, indent = 4)
    ))