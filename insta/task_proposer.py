from insta.task_parsers import (
    TASK_PARSERS
)

from insta.utils import (
    safe_call,
    BrowserStatus
)

from insta.configs.task_proposer_config import (
    TaskProposerConfig,
    DEFAULT_TASK_PROPOSER_CONFIG,
    BrowserTaskProposal
)

from typing import List, Callable
from transformers import AutoTokenizer

import openai


NULL_TASK_PROPOSAL = BrowserTaskProposal(
    proposed_task = None,
    task_is_feasible = None,
    estimated_difficulty = None,
    estimated_steps = None,
    response = None,
    matched_response = None
)


class BrowserTaskProposer(Callable):
    """Defines an LLM Task Proposer for LLM agents operating a web browser,
    served via the OpenAI API---local LLMs can be served using vLLM, 
    and proprietary LLMs can be accessed directly through the OpenAI API.

    Attributes: 

    config: TaskProposerConfig
        The configuration for the proposer, which includes the tokenizer
        to use, the client to use, and the generation kwargs to use,
        refer to insta/configs/task_proposer_config.py for more information.

    tokenizer: AutoTokenizer
        The tokenizer to use for encoding and decoding text, which is
        used for truncating observation text to a max length.

    task_parser: TaskParser
        The task proposal parser for parsing responses from the LLM into a
        dictionary containing a proposer task, and estimates of
        feasibility, difficulty, and steps to completion.

    llm_client: openai.OpenAI
        The OpenAI client for querying the LLM, provides a standard
        interface for connecting to a variety of LLMs, including
        local LLMs served via vLLM, GPT, and Gemini models
    
    """

    def __init__(self, config: TaskProposerConfig = DEFAULT_TASK_PROPOSER_CONFIG,
                 task_parser: str = "json"):
        """Defines an LLM Judge for evaluating agents operating a web browser,
        served via the OpenAI API---local LLMs can be served using vLLM, 
        and proprietary LLMs can be accessed directly through the OpenAI API.

        Arguments:

        config: TaskProposerConfig
            The configuration for the proposer, which includes the tokenizer
            to use, the client to use, and the generation kwargs to use,
            refer to insta/configs/task_proposer_config.py for more information.

        task_parser: TaskParser
            The task proposal parser for parsing responses from the LLM into a
            dictionary containing a proposer task, and estimates of
            feasibility, difficulty, and steps to completion.
        
        """

        super(BrowserTaskProposer, self).__init__()

        self.config = config

        self.task_parser = TASK_PARSERS[
            task_parser
        ]()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer
        )

        self.llm_client = openai.OpenAI(
            **self.config.client_kwargs
        )

    def get_task_proposal(
        self, observations: List[List[str]], 
        actions: List[List[str]],
        judgments: List[str],
        instructions: List[str],
        target_url: str,
        last_judgments: int = 5,
        last_tasks: int = 5,
        last_trajectories: int = 1,
        last_actions: int = 5,
        last_obs: int = 1
    ) -> BrowserTaskProposal | BrowserStatus:
        """Queries the LLM to propose a task for the agent to complete
        given previous attempts, and evaluations.

        Arguments:

        observations: List[List[str]]
            The current webpage processed into an agent-readible format,
            such as the markdown format used with InSTA.

        actions: List[List[str]]
            The previous actions the agent has taken in the browser,
            typically the raw LLM action output.

        judgments: List[str]
            An estimate of the agent's performance on the assigned task,
            typically the raw LLM judge output.

        instructions: List[str]
            The instruction to provide to the agent, such as a question
            or a command to execute on the web.

        target_url: str
            Propose tasks for the target URL.

        last_judgments: int
            The number of judgments to include in the context.

        last_tasks: int
            The number of tasks to include in the context.

        last_trajectories: int
            The number of trajectories to include in the context.

        last_actions: int
            The number of actions to include in the context.

        last_obs: int
            The number of observations to include in the context.

        Returns:

        BrowserTaskProposal | BrowserStatus
            The next assigned task for the agent, or NULL_TASK_PROPOSAL
            if the task failed to parse from the LLM output.

        """
        
        messages = self.get_context(
            observations = observations,
            actions = actions,
            judgments = judgments,
            instructions = instructions,
            target_url = target_url,
            last_judgments = last_judgments,
            last_tasks = last_tasks,
            last_trajectories = last_trajectories,
            last_actions = last_actions,
            last_obs = last_obs
        )

        completion = self.llm_client.chat.completions.create(
            messages = messages,
            **self.config.generation_kwargs
        )

        return self.task_parser.parse_task(
            completion.choices[0]
            .message.content
        )

    def __call__(
        self, observations: List[List[str]], 
        actions: List[List[str]],
        judgments: List[str],
        instructions: List[str],
        target_url: str
    ) -> BrowserTaskProposal | None:
        """Queries the LLM to propose a task for the agent to complete
        given previous attempts, and evaluations.

        Arguments:

        observations: List[List[str]]
            The current webpage processed into an agent-readible format,
            such as the markdown format used with InSTA.

        actions: List[List[str]]
            The previous actions the agent has taken in the browser,
            typically the raw LLM action output.

        judgments: List[str]
            An estimate of the agent's performance on the assigned task,
            typically the raw LLM judge output.

        instructions: List[str]
            The instruction to provide to the agent, such as a question
            or a command to execute on the web.

        target_url: str
            Propose tasks for the target URL.

        Returns:

        BrowserTaskProposal | BrowserStatus
            The next assigned task for the agent, or NULL_TASK_PROPOSAL
            if the task failed to parse from the LLM output.
        
        """

        task_proposal = safe_call(
            self.get_task_proposal,
            observations = observations,
            actions = actions,
            judgments = judgments,
            instructions = instructions,
            target_url = target_url,
            last_judgments = self.config.last_judgments,
            last_tasks = self.config.last_tasks,
            last_trajectories = self.config.last_trajectories,
            last_actions = self.config.last_actions,
            last_obs = self.config.last_obs,
            catch_errors = self.config.catch_errors,
            max_errors = self.config.max_errors,
            log_errors = self.config.log_errors
        )

        if task_proposal is BrowserStatus.ERROR:

            return NULL_TASK_PROPOSAL

        return task_proposal
    
    @property
    def system_prompt(self) -> str:

        return self.task_parser.system_prompt
    
    @property
    def user_prompt_template(self) -> str:

        return self.task_parser.user_prompt_template

    def get_user_prompt(
        self, observations: List[List[str]], 
        actions: List[List[str]],
        judgments: List[str],
        instructions: List[str],
        target_url: str,
        last_judgments: int = 5,
        last_tasks: int = 5,
        last_trajectories: int = 1,
        last_actions: int = 5,
        last_obs: int = 1
    ) -> List[dict]:
        """Builds the user prompt for querying the LLM to propose a task,
        and selects the N most recent trajectories, and includes the 
        last M actions, observations, and judgments.

        Arguments:

        observations: List[List[str]]
            The current webpage processed into an agent-readible format,
            such as the markdown format used with InSTA.

        actions: List[List[str]]
            The previous actions the agent has taken in the browser,
            typically the raw LLM action output.

        judgments: List[str]
            An estimate of the agent's performance on the assigned task,
            typically the raw LLM judge output.

        instructions: List[str]
            The instruction to provide to the agent, such as a question
            or a command to execute on the web.

        target_url: str
            Propose tasks for the target URL.

        last_judgments: int
            The number of judgments to include in the context.

        last_tasks: int
            The number of tasks to include in the context.

        last_trajectories: int
            The number of trajectories to include in the context.

        last_actions: int
            The number of actions to include in the context.

        last_obs: int
            The number of observations to include in the context.

        Returns:

        List[dict]
            The user prompt for querying the LLM to propose a task.
        
        """

        trajectory_outputs = []

        for trajectory_id, (
            trajectory_observations,
            trajectory_actions,
            trajectory_judgment,
            trajectory_instruction
        ) in enumerate(zip(
            observations,
            actions,
            judgments,
            instructions
        )):

            trajectories_left = (
                len(observations) - 
                trajectory_id - 1
            )

            if trajectories_left < last_tasks:

                trajectory_outputs.append(
                    "## {} Task:\n\n{}".format(
                    "Previous" 
                    if trajectories_left > 0 else 
                    "Last",
                    trajectory_instruction
                ))

            for trajectory_step, (
                observation,
                action
            ) in enumerate(zip(
                trajectory_observations,
                trajectory_actions
            )):

                observation = self.tokenizer.encode(
                    observation,
                    max_length = self.config.max_obs_tokens,
                    truncation = True
                )

                observation = self.tokenizer.decode(
                    observation,
                    skip_special_tokens = True
                )

                time_left = (
                    len(trajectory_observations) - 
                    trajectory_step - 1
                )

                if trajectories_left < last_trajectories \
                        and time_left < last_obs:

                    trajectory_outputs.append(
                        "## {} Webpage:\n\n{}".format(
                        "Previous" 
                        if time_left > 0 else 
                        "Last",
                        observation
                    ))

                if trajectories_left < last_trajectories \
                        and time_left < last_actions:
        
                    trajectory_outputs.append(
                        "## {} Action:\n\n{}".format(
                        "Previous" 
                        if time_left > 0 else 
                        "Last",
                        action
                    ))

            if trajectories_left < last_judgments:

                trajectory_outputs.append(
                    "## {} Performance Review:\n\n{}".format(
                    "Previous" 
                    if trajectories_left > 0 else 
                    "Last",
                    trajectory_judgment
                ))

        annotations = "\n\n".join(
            trajectory_outputs
        )
        
        return self.user_prompt_template.format(
            annotations = annotations,
            target_url = target_url
        )

    def get_context(
        self, observations: List[List[str]], 
        actions: List[List[str]],
        judgments: List[str],
        instructions: List[str],
        target_url: str,
        last_judgments: int = 5,
        last_tasks: int = 5,
        last_trajectories: int = 1,
        last_actions: int = 5,
        last_obs: int = 1
    ) -> List[dict]:
        """Construct a series of messages for querying an LLM via the
        OpenAI API to produce a judgment that estimates the
        performance of a web navigation agent.

        Arguments:

        observations: List[List[str]]
            The current webpage processed into an agent-readible format,
            such as the markdown format used with InSTA.

        actions: List[List[str]]
            The previous actions the agent has taken in the browser,
            typically the raw LLM action output.

        judgments: List[str]
            An estimate of the agent's performance on the assigned task,
            typically the raw LLM judge output.

        instructions: List[str]
            The instruction to provide to the agent, such as a question
            or a command to execute on the web.

        target_url: str
            Propose tasks for the target URL.

        last_judgments: int
            The number of judgments to include in the context.

        last_tasks: int
            The number of tasks to include in the context.

        last_trajectories: int
            The number of trajectories to include in the context.

        last_actions: int
            The number of actions to include in the context.

        last_obs: int
            The number of observations to include in the context.

        Returns:

        List[dict]
            The context for querying the LLM to propose a task.
        
        """

        user_prompt_str = self.get_user_prompt(
            observations = observations,
            actions = actions,
            judgments = judgments,
            instructions = instructions,
            target_url = target_url,
            last_judgments = last_judgments,
            last_tasks = last_tasks,
            last_trajectories = last_trajectories,
            last_actions = last_actions,
            last_obs = last_obs
        )

        system_prompt = {
            "role": "system",
            "content": self.system_prompt
        }

        user_prompt = {
            "role": "user",
            "content": user_prompt_str
        }

        return [
            system_prompt,
            user_prompt
        ]
