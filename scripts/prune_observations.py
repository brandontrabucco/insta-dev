from multiprocessing import Pool

import glob
import json
import tqdm
import argparse

import os


STYLE_KEYS_TO_KEEP = [
    'display'
]


def prune_observations(observations_file: str) -> None:
    """Reduce the size of the computed styles in the observations file
    by removing keys that are not needed.

    Arguments:

    observations_file: str
        The path to the observations file to prune.

    """
        
    with open(observations_file, 'r') as file:

        observations = json.load(file)

    for observation in observations:

        metadata = observation["metadata"]

        if metadata is None: continue

        for backend_node_id in metadata.keys():

            computed_style = metadata[backend_node_id]["computed_style"]

            computed_style_keys = list(computed_style.keys())

            for key in computed_style_keys:

                if key not in STYLE_KEYS_TO_KEEP:

                    del computed_style[key]

    with open(observations_file, 'w') as file:

        json.dump(
            observations,
            file,
            indent = 4
        )

    return observations_file


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = 'Check stats of agent trajectories'
    )

    parser.add_argument(
        '--observations_dir',
        type = str,
        default = 'data/observations',
        help = 'Directory containing observation files'
    )

    parser.add_argument(
        '--num_processes',
        type = int,
        default = 32,
        help = 'Number of processes to use'
    )

    args = parser.parse_args()

    observations_file_pattern = os.path.join(
        args.observations_dir, '*.json'
    )

    all_observations_files = glob.glob(observations_file_pattern)

    progress_bar = tqdm.tqdm(
        desc = 'Pruning observation files',
        dynamic_ncols = True,
        total = len(all_observations_files)
    )

    with Pool(args.num_processes) as pool:

        for observations_file in pool.imap_unordered(
                prune_observations, all_observations_files):
            
            progress_bar.update(1)