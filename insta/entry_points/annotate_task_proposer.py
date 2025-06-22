from insta import (
    get_task_proposer_config,
    TaskProposerConfig,
    BrowserTaskProposer,
    DEFAULT_TASK_PROPOSER_CONFIG,
    NULL_TASK_PROPOSAL
)

from insta.pipeline import (
    TASK_PROPOSER_EXPLORATION_TEMPLATE,
    TASK_PROPOSER_STEPS_TEMPLATE,
    TASK_PROPOSER_CRITERIA_TEMPLATE,
)

from insta.entry_points.args import (
    add_task_proposer_llm_args,
    add_task_proposer_name_args,
    add_task_proposer_prompt_args,
    add_task_proposer_sampling_args,
    add_judge_name_args,
    add_data_args,
    add_parallel_args,
    add_annotate_args,
    set_annotate_mode
)

from insta.entry_points.insta_pipeline import (
    get_task_proposer_config_from_cli,
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


DEFAULT_AGENT_RESPONSE_KEY = "response"
DEFAULT_JUDGE_RESPONSE_KEY = "response"

DEFAULT_STEPS = []
DEFAULT_CRITERIA = []


def query_task_proposer(
        example_id: int, dataset: Dataset,
        task_proposer_config: TaskProposerConfig = 
        DEFAULT_TASK_PROPOSER_CONFIG,
        observations_dir: str = None,
        actions_dir: str = None,
        judgments_dir: str = None,
        task_proposals_dir: str = None,
        agent_response_key: str = DEFAULT_AGENT_RESPONSE_KEY,
        judge_response_key: str = DEFAULT_JUDGE_RESPONSE_KEY,
        add_steps_to_task_proposer: bool = True,
        add_criteria_to_task_proposer: bool = True,
        skip_finished: bool = False):
    """Query the task proposer to annotate a single example from the dataset,
    and save the proposed task to the task proposals directory.

    Arguments:

    example_id: int
        ID of the example to annotate.

    dataset: Dataset
        Hugginggface dataset of websites, and task metadata.

    task_proposer_config: TaskProposerConfig
        Configuration for the task proposer, including the LLM and prompt.

    observations_dir: str
        Directory where the observations are stored.

    actions_dir: str
        Directory where the actions are stored.

    judgments_dir: str
        Directory where the judgments are stored.

    task_proposals_dir: str
        Directory where the task proposals will be saved.

    agent_response_key: str
        Key in action JSON for the agent's response.

    judge_response_key: str
        Key in judgment JSON for the judge's response.

    add_steps_to_task_proposer: bool
        Whether to add steps to the task proposer instruction.

    add_criteria_to_task_proposer: bool
        Whether to add criteria to the task proposer instruction.

    skip_finished: bool
        Whether to skip examples that have already been assigned tasks.

    Returns:

    identifier: str or None
        Identifier of the example if the judgment was created or None
        if the example is invalid or already judged.

    """

    example_dict = dataset[example_id]

    website = example_dict.get(
        "website", example_dict.get("domain")
    )

    identifier = example_dict.get(
        "identifier", website
    )

    instruction = example_dict.get(
        "instruction", example_dict.get(
            "task", TASK_PROPOSER_EXPLORATION_TEMPLATE.format(
                website = website
            )
        )
    )

    task_proposer_instruction = example_dict.get(
        "task_proposer_instruction", example_dict.get(
            "task_proposer_task", instruction
        )
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

    if add_steps_to_task_proposer and len(steps) > 0:

        task_proposer_instruction = TASK_PROPOSER_STEPS_TEMPLATE.format(
            instruction = task_proposer_instruction,
            steps = format_steps
        )

    if add_criteria_to_task_proposer and len(criteria) > 0:

        task_proposer_instruction = TASK_PROPOSER_CRITERIA_TEMPLATE.format(
            instruction = task_proposer_instruction,
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

    input_judgment_path = os.path.join(
        judgments_dir,
        "{}.json".format(identifier)
    )

    output_task_path = os.path.join(
        task_proposals_dir,
        "{}.json".format(identifier)
    )

    valid_example = (
        os.path.exists(input_actions_path)
        and os.path.exists(input_observations_path)
        and os.path.exists(input_judgment_path)
        and not (skip_finished and os.path.exists(output_task_path))
    )

    if not valid_example:

        return None

    with open(input_observations_path, "r") as file:
        
        try: observations = json.load(file)

        except: return None

    with open(input_actions_path, "r") as file:
        
        try: actions = json.load(file)

        except: return None

    with open(input_judgment_path, "r") as file:

        try: judgment = json.load(file)

        except: return None
    
    task_proposer = BrowserTaskProposer(
        config = task_proposer_config
    )

    task_proposal = task_proposer(
        instruction = task_proposer_instruction,
        website = website,
        observations = [
            x["processed_text"]
            for x in observations
        ],
        actions = [
            x[agent_response_key]
            for x in actions
        ],
        judgment = (
            judgment[
                judge_response_key
            ]
        ),
    )

    invalid_task_proposal = (
        task_proposal is None or 
        task_proposal == NULL_TASK_PROPOSAL
    )

    if invalid_task_proposal:

        return None

    task_proposal = {
        "proposed_task": task_proposal.proposed_task,
        "steps": task_proposal.steps,
        "criteria": task_proposal.criteria,
        "response": task_proposal.response,
        "matched_response": task_proposal.matched_response,
    }

    with open(output_task_path, "w") as file:
        
        json.dump(
            task_proposal,
            file,
            indent = 4
        )

    return identifier


def annotate_task_proposer_from_cli(args: argparse.Namespace):
    """Annotate with the task proposer from the command line arguments, refer to
    the command line arguments in insta.args.

    Arguments:

    args: argparse.Namespace
        The command line arguments for the insta pipeline.

    """

    set_annotate_mode(args)

    task_proposer_config = get_task_proposer_config_from_cli(
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
        data_dirs["task_proposals_dir"],
        exist_ok = True
    )

    progress_bar = tqdm.tqdm(
        desc = "Processing",
        dynamic_ncols = True,
        total = len(out_dataset_ids),
    )

    worker_fn = partial(
        query_task_proposer, dataset = dataset,
        task_proposer_config = task_proposer_config,
        observations_dir = data_dirs["observations_dir"],
        actions_dir = data_dirs["actions_dir"],
        judgments_dir = data_dirs["judgments_dir"],
        task_proposals_dir = data_dirs["task_proposals_dir"],
        agent_response_key = args.agent_response_key,
        judge_response_key = args.judge_response_key,
        add_steps_to_task_proposer = args.add_steps_to_task_proposer,
        add_criteria_to_task_proposer = args.add_criteria_to_task_proposer,
        skip_finished = args.skip_finished,
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


def start_annotate_task_proposer():
    """Annotate trajectories with the provided configurations, refer to
    the command line arguments in insta.args.

    """

    parser = argparse.ArgumentParser(
        description = "Annotate trajectories with the task proposer.",
    )

    parser = add_data_args(parser)
    parser = add_parallel_args(parser)
    parser = add_annotate_args(parser)
    
    parser = add_judge_name_args(parser)

    parser = add_task_proposer_llm_args(parser)
    parser = add_task_proposer_name_args(parser)
    parser = add_task_proposer_prompt_args(parser)
    parser = add_task_proposer_sampling_args(parser)

    args = parser.parse_args()

    annotate_task_proposer_from_cli(
        args = args
    )


if __name__ == "__main__":

    start_annotate_task_proposer()