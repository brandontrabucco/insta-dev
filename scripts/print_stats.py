from insta import VALUE_KEYS
import numpy as np

import argparse
import json
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

    all_judgments = []
    all_actions = []

    for judgment_file in os.listdir(judgments_dir):

        identifier = judgment_file.replace(
            ".json", ""
        )

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

        if not valid_example:
            
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

        if data_collection_error:
            
            continue

        all_judgments.append(judgment)
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

    print("Average length of trajectories: {:0.2f}\n".format(
        average_num_actions
    ))

    trajectories_with_stop = [
        actions[-1]['function_calls'][0]['dotpath'] == 'stop'
        for actions in all_actions
    ]

    print("Fraction trajectories that stop: {:0.2f}".format(
        sum(trajectories_with_stop)
        / len(trajectories_with_stop)
    ))

    length_trajectories_with_stop = [
        len(actions) for actions in all_actions
        if actions[-1]['function_calls'][0]['dotpath'] == 'stop'
    ]

    print("Average length of trajectories that stop: {:0.2f}\n".format(
        sum(length_trajectories_with_stop)
        / len(length_trajectories_with_stop)
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

    for threshold in [0.5, 0.7, 1.0]:

        cdf_scores = {
            value_key: get_cdf_score(value_key, threshold)
            for value_key in VALUE_KEYS
        }
    
        print('Judge(Success) {} {}: {}\n'.format(
            "=" if threshold == 1 else ">",
            threshold, json.dumps(cdf_scores, indent = 4)
        ))