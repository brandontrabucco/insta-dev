from insta import (
    get_judge_config,
    JudgeConfig,
    BrowserJudge,
    DEFAULT_JUDGE_CONFIG,
    NULL_JUDGMENT
)

from insta.pipeline import (
    JUDGE_EXPLORATION_TEMPLATE,
    JUDGE_STEPS_TEMPLATE,
    JUDGE_CRITERIA_TEMPLATE,
)

from insta.entry_points.args import (
    add_judge_llm_args,
    add_judge_name_args,
    add_judge_prompt_args,
    add_judge_sampling_args,
    add_data_args,
    add_parallel_args,
    add_annotate_args,
    set_annotate_mode
)

from insta.entry_points.insta_pipeline import (
    get_judge_config_from_cli,
    get_data_dirs_from_cli,
    get_dataset_from_cli,
)

from multiprocessing import Pool
from functools import partial

from datasets import (
    load_dataset,
    Dataset
)

import argparse
import random

import tqdm
import json
import os

from insta.configs.judge_config import (
    VALUE_KEYS
)


DEFAULT_AGENT_RESPONSE_KEY = "response"

DEFAULT_STEPS = []
DEFAULT_CRITERIA = []


def query_judge(
        example_id: int, dataset: Dataset,
        judge_config: JudgeConfig = DEFAULT_JUDGE_CONFIG,
        observations_dir: str = None,
        actions_dir: str = None,
        judgments_dir: str = None,
        agent_response_key: str = DEFAULT_AGENT_RESPONSE_KEY,
        add_steps_to_judge: bool = True,
        add_criteria_to_judge: bool = True,
        skip_finished: bool = False):
    """Query the judge to annotate a single example from the dataset,
    and save the judgment to the judgments directory.

    Arguments:

    example_id: int
        ID of the example to annotate.

    dataset: Dataset
        Hugginggface dataset of websites, and task metadata.

    judge_config: JudgeConfig
        Configuration for the judge, including the LLM and prompt.

    observations_dir: str
        Directory where the observations are stored.

    actions_dir: str
        Directory where the actions are stored.

    judgments_dir: str
        Directory where the judgments will be saved.

    agent_response_key: str
        Key in action JSON for the agent's response.

    add_steps_to_judge: bool
        Whether to add steps to the judge instruction.

    add_criteria_to_judge: bool
        Whether to add criteria to the judge instruction.

    skip_finished: bool
        Whether to skip examples that have already been judged.

    Returns:

    identifier: str or None
        Identifier of the example if the judgment was created or None
        if the example is invalid or already judged.

    """

    example_dict = dataset[example_id]

    domain = example_dict.get(
        "website", example_dict.get("domain")
    )

    instruction = example_dict.get(
        "instruction", example_dict.get(
            "task", JUDGE_EXPLORATION_TEMPLATE.format(
                website = website
            )
        )
    )

    judge_instruction = example_dict.get(
        "judge_instruction", example_dict.get(
            "judge_task", instruction
        )
    )

    identifier = example_dict.get(
        "identifier", domain
    )

    steps = example_dict.get(
        "steps", DEFAULT_STEPS
    )

    criteria = example_dict.get(
        "criteria", DEFAULT_CRITERIA
    )

    format_steps = "\n".join(
        "{n}. {part}".format(n = idx + 1, part = part)
        for idx, part in enumerate(steps)
    )

    format_criteria = "\n".join(
        "{n}. {part}".format(n = idx + 1, part = part)
        for idx, part in enumerate(criteria)
    )

    if add_steps_to_judge and len(steps) > 0:

        judge_instruction = JUDGE_STEPS_TEMPLATE.format(
            instruction = judge_instruction,
            steps = format_steps
        )

    if add_criteria_to_judge and len(criteria) > 0:

        judge_instruction = JUDGE_CRITERIA_TEMPLATE.format(
            instruction = judge_instruction,
            criteria = format_criteria
        )

    input_observations_path = os.path.join(
        observations_dir,
        "{}.json".format(identifier)
    )

    input_actions_path = os.path.join(
        actions_dir,
        "{}.json".format(identifier)
    )

    output_judgment_path = os.path.join(
        judgments_dir,
        "{}.json".format(identifier)
    )

    valid_example = (
        os.path.exists(input_actions_path)
        and os.path.exists(input_observations_path)
        and not (skip_finished and os.path.exists(output_judgment_path))
    )

    if not valid_example:

        return None

    with open(input_observations_path, "r") as file:
        
        try: observations = json.load(file)

        except: return None

    with open(input_actions_path, "r") as file:
        
        try: actions = json.load(file)

        except: return None
    
    judge = BrowserJudge(
        config = judge_config
    )

    judgment = judge(
        observations = [
            x["processed_text"]
            for x in observations
        ],
        actions = [
            x[agent_response_key]
            for x in actions
        ],
        instruction = judge_instruction,
    )

    invalid_judgment = (
        judgment is None or 
        judgment == NULL_JUDGMENT
    )

    if invalid_judgment:

        return None

    judgment_values = {
        key: judgment.values.get(key)
        for key in VALUE_KEYS
    }

    if any([val is None for val in judgment_values.values()]):

        return None

    judgment = {
        **judgment_values,
        "response": judgment.response,
        "matched_response": judgment.matched_response,
    }

    with open(output_judgment_path, "w") as file:
        
        json.dump(
            judgment,
            file,
            indent = 4
        )

    return identifier


def annotate_judge_from_cli(args: argparse.Namespace):
    """Annotate with the judge from the command line arguments, refer to
    the command line arguments in insta.args.

    Arguments:

    args: argparse.Namespace
        The command line arguments for the insta pipeline.

    """

    set_annotate_mode(args)

    judge_config = get_judge_config_from_cli(
        args = args
    )

    data_dirs = get_data_dirs_from_cli(
        args = args
    )

    dataset = get_dataset_from_cli(
        args = args
    )

    dataset_ids = list(range(len(dataset)))

    random.seed(args.seed)
    random.shuffle(dataset_ids)

    out_dataset_ids = []

    for agent_rank in range(
            args.rank * args.num_workers,
            (args.rank + 1) * args.num_workers):

        out_dataset_ids.extend(dataset_ids[
            agent_rank::args.num_workers * args.world_size
        ])

    os.makedirs(
        data_dirs["judgments_dir"],
        exist_ok = True
    )

    progress_bar = tqdm.tqdm(
        desc = "Processing",
        dynamic_ncols = True,
        total = len(out_dataset_ids),
    )

    worker_fn = partial(
        query_judge, dataset = dataset,
        judge_config = judge_config,
        observations_dir = data_dirs["observations_dir"],
        actions_dir = data_dirs["actions_dir"],
        judgments_dir = data_dirs["judgments_dir"],
        agent_response_key = args.agent_response_key,
        add_steps_to_judge = args.add_steps_to_judge,
        add_criteria_to_judge = args.add_criteria_to_judge,
        skip_finished = args.skip_finished
    )
    
    with Pool(processes = args.num_workers) as pool:

        for identifier in pool.imap_unordered(
            worker_fn,
            out_dataset_ids
        ):
            
            progress_bar.update()

            if identifier is not None:

                progress_bar.set_description(
                    "Processing {}"
                    .format(identifier)
                )

def start_annotate_judge():
    """Annotate trajectories with the provided configurations, refer to
    the command line arguments in insta.args.

    """

    parser = argparse.ArgumentParser(
        description = "Annotate trajectories with the judge.",
    )

    parser = add_data_args(parser)
    parser = add_parallel_args(parser)
    parser = add_annotate_args(parser)

    parser = add_judge_llm_args(parser)
    parser = add_judge_name_args(parser)
    parser = add_judge_prompt_args(parser)
    parser = add_judge_sampling_args(parser)

    args = parser.parse_args()

    annotate_judge_from_cli(
        args = args
    )


if __name__ == "__main__":

    start_annotate_judge()