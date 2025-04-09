from insta.utils import (
    VALUE_KEYS
)

from collections import defaultdict
from tqdm import tqdm

import glob
import json
import argparse
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = 'Check stats of agent trajectories'
    )

    parser.add_argument(
        '--judgments_pattern',
        type = str,
        default = 'qwen-1.5b-grpo-n0-rollouts/*/judgments/*.json',
        help = 'Directory containing judgment files'
    )

    args = parser.parse_args()

    task_to_judgments = defaultdict(list)

    for judgment_file in glob.glob(args.judgments_pattern):

        website_name = (
            os.path.basename(judgment_file)
            .removesuffix('.json')
        )

        with open(judgment_file, 'r') as f:
            judgment = json.load(f)

        task_to_judgments[website_name].append(judgment)

    all_judgments = [
        max(
            judgments, key = (
                lambda x: x['success'] or 0.0
            )
        )
        for x, judgments in 
        task_to_judgments.items()
    ]

    fraction_ge_0_5 = {
        key: sum(
            (judgment[key] or 0) > 0.5
            for judgment in all_judgments
        ) / len(all_judgments)
        for key in VALUE_KEYS
    }

    print('Fraction conf > 0.5: {}\n'.format(
        json.dumps(fraction_ge_0_5, indent = 4)
    ))
