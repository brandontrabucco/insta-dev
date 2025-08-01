from datasets import load_dataset

from insta import (
    get_browser_config,
    get_agent_config,
    get_judge_config,
    get_task_proposer_config,
    InstaPipeline
)

from insta.entry_points.args import (
    add_agent_llm_args,
    add_agent_prompt_args,
    add_agent_sampling_args,
    add_judge_llm_args,
    add_judge_name_args,
    add_judge_prompt_args,
    add_judge_sampling_args,
    add_task_proposer_llm_args,
    add_task_proposer_name_args,
    add_task_proposer_prompt_args,
    add_task_proposer_sampling_args,
    add_data_args,
    add_parallel_args,
    add_playwright_args,
    add_pipeline_args
)

import argparse
import os


def get_agent_config_from_cli(args: argparse.Namespace):
    """Load the agent from the command line arguments, refer to
    the command line arguments in insta.args.

    Arguments:

    args: argparse.Namespace
        The command line arguments for the insta pipeline.

    Returns:

    agent_config: AgentConfig

    """

    agent_client_type = "openai"

    agent_client_kwargs = {
        "api_key": args.agent_api_key,
        "base_url": args.agent_llm_endpoint
    }

    agent_generation_kwargs = {
        "model": args.agent_model_name,
        "max_tokens": 1024,
        "top_p": args.agent_top_p,
        "temperature": args.agent_temperature,
        "extra_body": {}
    }

    if args.agent_reasoning_effort:

        agent_generation_kwargs.update({
            "reasoning_effort": 
            args.agent_reasoning_effort
        })

    if args.agent_disable_thinking_chat_template:

        if "chat_template_kwargs" not in agent_generation_kwargs["extra_body"]:

            agent_generation_kwargs["extra_body"].update({
                "chat_template_kwargs": {}
            })

        agent_generation_kwargs["extra_body"]["chat_template_kwargs"].update({
            "enable_thinking": False
        })

    if args.agent_top_k is not None:

        agent_generation_kwargs["extra_body"].update({
            "top_k": args.agent_top_k
        })

    agent_config = get_agent_config(
        client_type = agent_client_type,
        client_kwargs = agent_client_kwargs,
        generation_kwargs = agent_generation_kwargs,
        agent_prompt = args.agent_prompt,
        max_obs_tokens = args.max_obs_tokens,
        last_obs = args.last_obs,
        log_errors = True,
    )

    return agent_config


def get_judge_config_from_cli(args: argparse.Namespace):
    """Load the judge from the command line arguments, refer to
    the command line arguments in insta.args.

    Arguments:

    args: argparse.Namespace
        The command line arguments for the insta pipeline.

    Returns:

    judge_config: JudgeConfig

    """

    judge_client_type = "openai"

    judge_client_kwargs = {
        "api_key": args.judge_api_key,
        "base_url": args.judge_llm_endpoint
    }

    judge_generation_kwargs = {
        "model": args.judge_model_name,
        "max_tokens": 1024,
        "top_p": args.judge_top_p,
        "temperature": args.judge_temperature,
        "extra_body": {}
    }

    if args.judge_reasoning_effort:

        judge_generation_kwargs.update({
            "reasoning_effort": 
            args.judge_reasoning_effort
        })

    if args.judge_disable_thinking_chat_template:

        if "chat_template_kwargs" not in judge_generation_kwargs["extra_body"]:

            judge_generation_kwargs["extra_body"].update({
                "chat_template_kwargs": {}
            })

        judge_generation_kwargs["extra_body"]["chat_template_kwargs"].update({
            "enable_thinking": False
        })

    if args.judge_top_k is not None:

        judge_generation_kwargs["extra_body"].update({
            "top_k": args.judge_top_k
        })

    judge_config = get_judge_config(
        client_type = judge_client_type,
        client_kwargs = judge_client_kwargs,
        generation_kwargs = judge_generation_kwargs,
        log_errors = True,
    )

    return judge_config


def get_task_proposer_config_from_cli(args: argparse.Namespace):
    """Load the task proposer from the command line arguments, refer to
    the command line arguments in insta.args.

    Arguments:

    args: argparse.Namespace
        The command line arguments for the insta pipeline.

    Returns:

    task_proposer_config: TaskProposerConfig

    """

    task_proposer_client_type = "openai"

    task_proposer_client_kwargs = {
        "api_key": args.task_proposer_api_key,
        "base_url": args.task_proposer_llm_endpoint
    }

    task_proposer_generation_kwargs = {
        "model": args.task_proposer_model_name,
        "max_tokens": 1024,
        "top_p": args.task_proposer_top_p,
        "temperature": args.task_proposer_temperature,
        "extra_body": {}
    }

    if args.task_proposer_reasoning_effort:

        task_proposer_generation_kwargs.update({
            "reasoning_effort": 
            args.task_proposer_reasoning_effort
        })

    if args.task_proposer_disable_thinking_chat_template:

        if "chat_template_kwargs" not in task_proposer_generation_kwargs["extra_body"]:

            task_proposer_generation_kwargs["extra_body"].update({
                "chat_template_kwargs": {}
            })

        task_proposer_generation_kwargs["extra_body"]["chat_template_kwargs"].update({
            "enable_thinking": False
        })

    if args.task_proposer_top_k is not None:

        task_proposer_generation_kwargs["extra_body"].update({
            "top_k": args.task_proposer_top_k
        })

    task_proposer_config = get_task_proposer_config(
        client_type = task_proposer_client_type,
        client_kwargs = task_proposer_client_kwargs,
        generation_kwargs = task_proposer_generation_kwargs,
        log_errors = True,
    )

    return task_proposer_config


def get_dataset_from_cli(args: argparse.Namespace):
    """Load the dataset specified in the command line arguments, refer to
    the command line arguments in insta.args.

    Arguments:

    args: argparse.Namespace
        The command line arguments for the insta pipeline.

    Returns:

    dataset: datasets.Dataset

    """

    dataset = load_dataset(
        args.dataset,
        split = args.dataset_split
    )

    if args.set_exploration_mode:

        dataset = dataset.remove_columns(list({
            "instruction", "task", "steps", "criteria"
        } & set(dataset.column_names)))

    return dataset


def get_data_dirs_from_cli(args: argparse.Namespace):
    """Get the data directories from the command line arguments, refer to
    the command line arguments in insta.args.

    Arguments:

    args: argparse.Namespace
        The command line arguments for the insta pipeline.

    Returns:

    input_data_dirs: dict
        A dictionary containing the directories for observations,
        screenshots, actions, and optional judgments, and task proposals.

    """

    observations_dir = os.path.join(
        args.input_data_dir,
        "observations"
    )

    screenshot_dir = os.path.join(
        args.input_data_dir,
        "screenshots"
    )

    actions_dir = os.path.join(
        args.input_data_dir,
        "actions"
    )

    judgments_dir = None

    if args.set_annotate_judge or args.set_annotate_task_proposer:

        judgments_dir = os.path.join(
            args.input_data_dir,
            args.judge_name
        )

    task_proposals_dir = None

    if args.set_annotate_task_proposer:

        task_proposals_dir = os.path.join(
            args.input_data_dir,
            args.task_proposer_name
        )

    return {
        "observations_dir": observations_dir,
        "screenshot_dir": screenshot_dir,
        "actions_dir": actions_dir,
        "judgments_dir": judgments_dir,
        "task_proposals_dir": task_proposals_dir
    }


def get_pipeline_from_cli(args: argparse.Namespace):
    """Build the InSTA pipeline from the command line arguments, refer to
    the command line arguments in insta.args.

    Arguments:

    args: argparse.Namespace
        The command line arguments for the insta pipeline.

    Returns:

    pipeline: insta.InstaPipeline

    """

    browser_config = get_browser_config(
        playwright_url = args.playwright_url,
        playwright_port = args.playwright_port
    )

    agent_config = get_agent_config_from_cli(
        args = args
    )

    judge_config = None

    if args.set_annotate_judge or args.set_annotate_task_proposer:

        judge_config = get_judge_config_from_cli(
            args = args
        )
        
    task_proposer_config = None

    if args.set_annotate_task_proposer:

        task_proposer_config = get_task_proposer_config_from_cli(
            args = args
        )

    data_dirs = get_data_dirs_from_cli(
        args = args
    )

    pipeline = InstaPipeline(
        browser_config = browser_config,
        agent_config = agent_config,
        judge_config = judge_config,
        task_proposer_config = task_proposer_config,
        seed = args.seed, rank = args.rank,
        world_size = args.world_size,
        observations_dir = data_dirs["observations_dir"],
        screenshot_dir = data_dirs["screenshot_dir"],
        actions_dir = data_dirs["actions_dir"],
        judgments_dir = data_dirs["judgments_dir"],
        task_proposals_dir = data_dirs["task_proposals_dir"],
        skip_finished = args.skip_finished,
        prune_observations = args.prune_observations,
        add_steps_to_agent = args.add_steps_to_agent,
        add_criteria_to_agent = args.add_criteria_to_agent,
        add_steps_to_judge = args.add_steps_to_judge,
        add_criteria_to_judge = args.add_criteria_to_judge,
        add_steps_to_task_proposer = args.add_steps_to_task_proposer,
        add_criteria_to_task_proposer = args.add_criteria_to_task_proposer,
        agent_response_key = args.agent_response_key,
        judge_response_key = args.judge_response_key,
        max_actions = args.max_actions,
    )

    return pipeline


def launch_pipeline_from_cli(args: argparse.Namespace):
    """Launch the InSTA pipeline from the command line arguments, refer to
    the command line arguments in insta.args.

    Arguments:

    args: argparse.Namespace
        The command line arguments for the insta pipeline.

    """

    pipeline = get_pipeline_from_cli(
        args = args
    )

    dataset = get_dataset_from_cli(
        args = args
    )

    pipeline.launch(
        dataset = dataset,
        num_agents = args.num_agents,
        playwright_workers = args.playwright_workers,
        return_trajectories = False
    )


def start_insta_pipeline():
    """Run the InSTA pipeline with the provided configurations, refer to
    the command line arguments in insta.args.

    """

    parser = argparse.ArgumentParser(
        description = "Run the InSTA pipeline."
    )

    parser = add_agent_llm_args(parser)
    parser = add_agent_prompt_args(parser)
    parser = add_agent_sampling_args(parser)
    
    parser = add_judge_llm_args(parser)
    parser = add_judge_name_args(parser)
    parser = add_judge_prompt_args(parser)
    parser = add_judge_sampling_args(parser)

    parser = add_task_proposer_llm_args(parser)
    parser = add_task_proposer_name_args(parser)
    parser = add_task_proposer_prompt_args(parser)
    parser = add_task_proposer_sampling_args(parser)

    parser = add_data_args(parser)
    parser = add_parallel_args(parser)
    parser = add_playwright_args(parser)
    parser = add_pipeline_args(parser)

    launch_pipeline_from_cli(
        args = parser.parse_args()
    )


if __name__ == "__main__":

    start_insta_pipeline()