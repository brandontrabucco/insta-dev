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

    parser.add_argument(
        '--dump_sites',
        type = str,
        default = None,
    )

    parser.add_argument(
        '--load_sites',
        type = str,
        default = None,
    )

    args = parser.parse_args()

    judgment_file_pattern = os.path.join(
        args.judgments_dir, '*.json'
    )

    all_judgments = []
    all_actions = []

    sites = set()

    if args.load_sites:

        with open(args.load_sites, 'r') as file:

            sites = set(json.load(file))

    for judgment_file in glob.glob(judgment_file_pattern):

        if args.load_sites:

            site = (
                os.path.basename(judgment_file)
                .removesuffix('.json')
            )

            if site not in sites:

                continue

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

        sites.add(
            os.path.basename(judgment_file)
            .removesuffix('.json')
        )

    if args.dump_sites:

        sites = list(sites)

        with open(args.dump_sites, 'w') as file:

            json.dump(
                sites,
                file,
                indent = 4
            )

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

    average_values = {
        key: sum(
            judgment.get(key) or 0.0 
            for judgment in all_judgments
        ) / len(all_judgments)
        for key in VALUE_KEYS
    }

    print('Average: {}\n'.format(
        json.dumps(average_values, indent = 4)
    ))

    fraction_eq_1 = {
        key: sum(
            judgment.get(key) == 1 
            for judgment in all_judgments
        ) / len(all_judgments)
        for key in VALUE_KEYS
    }

    print('Fraction conf = 1: {}\n'.format(
        json.dumps(fraction_eq_1, indent = 4)
    ))

    fraction_ge_0_5 = {
        key: sum(
            (judgment.get(key) or 0) > 0.5
            for judgment in all_judgments
        ) / len(all_judgments)
        for key in VALUE_KEYS
    }

    print('Fraction conf > 0.5: {}\n'.format(
        json.dumps(fraction_ge_0_5, indent = 4)
    ))
