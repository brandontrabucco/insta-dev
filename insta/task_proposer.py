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

from typing import Tuple, List, Callable
from transformers import AutoTokenizer

from vllm import (
    LLM, SamplingParams
)

import openai


NULL_TASK_PROPOSAL = BrowserTaskProposal(
    proposed_task = None,
    steps = None,
    criteria = None,
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

    observations: List[List[str]]
    actions: List[List[str]]
    judgments: List[str]
    instructions: List[str]
    task_proposals: List[str]

    def __init__(self, config: TaskProposerConfig = DEFAULT_TASK_PROPOSER_CONFIG):
        """Defines an LLM Task Proposer for LLM agents operating a web browser,
        served via the OpenAI API---local LLMs can be served using vLLM, 
        and proprietary LLMs can be accessed directly through the OpenAI API.

        Arguments:

        config: TaskProposerConfig
            The configuration for the proposer, which includes the tokenizer
            to use, the client to use, and the generation kwargs to use,
            refer to insta/configs/task_proposer_config.py for more information.
        
        """

        super(BrowserTaskProposer, self).__init__()

        self.config = config

        self.task_parser = TASK_PARSERS[
            self.config.task_parser
        ]()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer
        )

        if self.config.client_type == "vllm":

            self.sampling_params = SamplingParams(
                **self.config.generation_kwargs
            )

            self.llm_client = LLM(
                **self.config.client_kwargs
            )

        elif self.config.client_type == "openai":

            self.llm_client = openai.OpenAI(
                **self.config.client_kwargs
            )

        self.reset()

    def reset(self) -> None:
        """Reset the context for the LLM Task Proposer, and remove previous 
        observations and actions, judgments, and instructions.
        
        """

        self.observations = []
        self.actions = []
        self.judgments = []
        self.instructions = []
        self.task_proposals = []

    def get_task_proposal(
        self, observations: List[List[str]], 
        actions: List[List[str]],
        judgments: List[str],
        instructions: List[str],
        task_proposals: List[str],
        target_url: str,
        last_judgments: int = 5,
        last_tasks: int = 5,
        last_trajectories: int = 1,
        last_actions: int = 5,
        last_obs: int = 5
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
            or a command to execute in the browser.

        task_proposals: List[str]
            Previous task proposals from the LLM.

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
        
        messages = self.get_prompts(
            observations = observations,
            actions = actions,
            judgments = judgments,
            instructions = instructions,
            task_proposals = task_proposals,
            target_url = target_url,
            last_judgments = last_judgments,
            last_tasks = last_tasks,
            last_trajectories = last_trajectories,
            last_actions = last_actions,
            last_obs = last_obs
        )

        if self.config.client_type == "vllm":

            response = self.llm_client.chat(
                messages = messages,
                sampling_params = self.sampling_params
            )[0].outputs[0].text

        elif self.config.client_type == "openai":

            response = self.llm_client.chat.completions.create(
                messages = messages,
                **self.config.generation_kwargs
            ).choices[0].message.content

        return self.task_parser.parse_task(
            response = response
        )

    def __call__(
        self, observations: List[str], 
        actions: List[str],
        judgment: str,
        instruction: str,
        target_url: str
    ) -> BrowserTaskProposal | None:
        """Queries the LLM to propose a task for the agent to complete
        given previous attempts, and evaluations.

        Arguments:

        observations: List[str]
            The current webpage processed into an agent-readible format,
            such as the markdown format used with InSTA.

        actions: List[str]
            The previous actions the agent has taken in the browser,
            typically the raw LLM action output.

        judgment: str
            An estimate of the agent's performance on the assigned task,
            typically the raw LLM judge output.

        instruction: str
            The instruction to provide to the agent, such as a question
            or a command to execute in the browser.

        target_url: str
            Propose tasks for the target URL.

        Returns:

        BrowserTaskProposal | BrowserStatus
            The next assigned task for the agent, or NULL_TASK_PROPOSAL
            if the task failed to parse from the LLM output.
        
        """

        self.push_trajectory(
            observations = observations,
            actions = actions,
            judgment = judgment,
            instruction = instruction
        )

        task_proposal = safe_call(
            self.get_task_proposal,
            observations = self.observations,
            actions = self.actions,
            judgments = self.judgments,
            instructions = self.instructions,
            task_proposals = self.task_proposals,
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

    def push_trajectory(
        self, observations: List[str],
        actions: List[str],
        judgment: str,
        instruction: str
    ):
        """Pushes the latest trajectory to the context.

        Arguments:

        observations: List[str]
            The current webpage processed into an agent-readible format,
            such as the markdown format used with InSTA.

        actions: List[str]
            The previous actions the agent has taken in the browser,
            typically the raw LLM action output.

        judgment: str
            An estimate of the agent's performance on the assigned task,
            typically the raw LLM judge output.

        instruction: str
            The instruction to provide to the agent, such as a question
            or a command to execute in the browser.

        """

        self.observations.append(observations)
        self.actions.append(actions)
        self.judgments.append(judgment)
        self.instructions.append(instruction)

    def push_task_proposal(self, response: str):
        """Add the specified agent response to the context, manual
        to handle cases where the agent response is not generated
        by the LLM, or is post-processed by the developer.

        Arguments:

        response: str
            The last response generated by the agent, which includes
            a chain of thought, and a task proposal.
        
        """

        self.task_proposals.append(response)

    def pop_trajectory(self) -> Tuple[List[str], List[str], str, str] | None:
        """If the last item in the context is a trajectory, then
        remove and return the last trajectory.

        Returns:

        Tuple[List[str], List[str], str, str] | None
            The last trajectory, removed from the context, or None if
            the last item is not a trajectory.
        
        """

        has_last_trajectory = (
            len(self.observations) > 0 and 
            len(self.actions) == len(self.observations) and
            len(self.judgments) == len(self.observations) and
            len(self.instructions) == len(self.observations) and
            len(self.task_proposals) == len(self.observations) 
        )

        if has_last_trajectory:

            observations = self.observations.pop()
            actions = self.actions.pop()
            judgment = self.judgments.pop()
            instruction = self.instructions.pop()
            
            return (
                observations,
                actions,
                judgment,
                instruction
            )

    def pop_task_proposal(self) -> str | None:
        """If the final item in the context is a task proposal, then
        pop the task from the context and return it.

        Returns:

        task_proposal: str | None
            The last task proposal, removed from the context, or None
            if the last item is not a task proposal.

        """

        has_last_task_proposal = (
            len(self.task_proposals) > 0 and
            len(self.task_proposals) == (len(self.observations) - 1)
        )

        if has_last_task_proposal:

            return self.task_proposals.pop()
    
    def get_context(self) -> Tuple[List[List[str]], List[List[str]], List[str], List[str], List[str]]:
        """Returns the current context for the proposer, which includes
        previous observations, and actions the agent has performed
        in the current browsing session.
        
        Returns:
        
        Tuple[List[List[str]], List[List[str]], List[str], List[str], List[str]]
            The entire context, which contains previous observations, and 
            actions the agent has performed in the browser.
            
        """
            
        return (
            self.observations,
            self.actions,
            self.judgments,
            self.instructions,
            self.task_proposals
        )
    
    def set_context(self, context: Tuple[List[List[str]], List[List[str]], List[str], List[str], List[str]]):
        """Replace the proposer context with the specified context, which
        includes target observations, and actions the agent has performed
        in a different browsing session.
        
        Arguments:
        
        Tuple[List[List[str]], List[List[str]], List[str], List[str], List[str]]
            An entire context, which contains previous observations, and 
            actions an agent has performed in the browser.
        
        """

        has_valid_context = (
            len(context) == 5 and
            all([isinstance(x, list) for x in context]) 
        )

        if not has_valid_context:

            raise ValueError(
                "Context must have the following type: "
                "Tuple[List[List[str]], List[List[str]], List[str], List[str], List[str]]"
            )
        
        self.observations = context[0]
        self.actions = context[1]
        self.judgments = context[2]
        self.instructions = context[3]
        self.task_proposals = context[4]

    @property
    def system_prompt(self) -> str:

        return self.task_parser.system_prompt
    
    @property
    def user_prompt_template(self) -> str:

        return self.task_parser.user_prompt_template

    def get_single_user_prompt(
        self, observations: List[str], 
        actions: List[str],
        judgment: str,
        instruction: str,
        target_url: str,
        trajectories_left: int = None,
        last_judgments: int = 5,
        last_tasks: int = 5,
        last_trajectories: int = 1,
        last_actions: int = 5,
        last_obs: int = 5
    ) -> str:
        """Builds the user prompt for querying the LLM to propose a task,
        and selects the N most recent trajectories, and includes the 
        last M actions, observations, and judgments.

        Arguments:

        observations: List[str]
            The current webpage processed into an agent-readible format,
            such as the markdown format used with InSTA.

        actions: List[str]
            The previous actions the agent has taken in the browser,
            typically the raw LLM action output.

        judgment: str
            An estimate of the agent's performance on the assigned task,
            typically the raw LLM judge output.

        instruction: str
            The instruction to provide to the agent, such as a question
            or a command to execute in the browser.

        target_url: str
            Propose tasks for the target URL.

        trajectories_left: int
            The number of trajectories left to include in the context.

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

        user_prompt: str
            The user prompt for task proposal.
        
        """

        trajectory_outputs = []

        if trajectories_left < last_tasks:

            trajectory_outputs.append(
                "## {} Task:\n\n{}".format(
                "Previous" 
                if trajectories_left > 0 else
                "Last",
                instruction
            ))

        for trajectory_step, (
            observation,
            action
        ) in enumerate(zip(
            observations,
            actions
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
                len(observations) - 
                trajectory_step - 1
            )

            include_observation = (
                trajectories_left < last_trajectories
                and time_left < last_obs
            )

            if include_observation:

                trajectory_outputs.append(
                    "## {} Webpage:\n\n{}".format(
                    "Previous" 
                    if time_left > 0 else 
                    "Last",
                    observation
                ))

            include_action = (
                trajectories_left < last_trajectories
                and time_left < last_actions
            )

            if include_action:
    
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
                judgment
            ))

        if len(trajectory_outputs) == 0:

            return None
        
        return self.user_prompt_template.format(
            annotations = "\n\n".join(trajectory_outputs),
            target_url = target_url
        )

    def get_user_prompts(
        self, observations: List[List[str]], 
        actions: List[List[str]],
        judgments: List[str],
        instructions: List[str],
        target_url: str,
        last_judgments: int = 5,
        last_tasks: int = 5,
        last_trajectories: int = 1,
        last_actions: int = 5,
        last_obs: int = 5
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
            or a command to execute in the browser.

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

        user_prompts: List[dict]
            User prompts for each step of task proposal.
        
        """

        user_prompts = []

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

            user_prompt_str = self.get_single_user_prompt(
                observations = trajectory_observations,
                actions = trajectory_actions,
                judgment = trajectory_judgment,
                instruction = trajectory_instruction,
                target_url = target_url,
                trajectories_left = trajectories_left,
                last_judgments = last_judgments,
                last_tasks = last_tasks,
                last_trajectories = last_trajectories,
                last_actions = last_actions,
                last_obs = last_obs
            )

            if user_prompt_str is not None:

                user_prompts.append({
                    "role": "user",
                    "content": user_prompt_str
                })

        return user_prompts

    def get_prompts(
        self, observations: List[List[str]], 
        actions: List[List[str]],
        judgments: List[str],
        instructions: List[str],
        task_proposals: List[str],
        target_url: str,
        last_judgments: int = 5,
        last_tasks: int = 5,
        last_trajectories: int = 1,
        last_actions: int = 5,
        last_obs: int = 5
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
            or a command to execute in the browser.

        task_proposals: List[str]
            Previous task proposals from the LLM.

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

        valid_arguments = (
            len(observations) == len(actions) and 
            len(observations) == len(judgments) and 
            len(observations) == len(instructions) and 
            len(task_proposals) == (len(observations) - 1)
        )

        if not valid_arguments:

            raise ValueError(
                "Invalid task proposer context."
            )

        system_prompt = {
            "role": "system",
            "content": self.system_prompt
        }

        *user_prompts, last_user_prompt = self.get_user_prompts(
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

        assistant_prompts = [{
            "role": "assistant",
            "content": task_proposal
        } for task_proposal in task_proposals]

        assistant_prompts = assistant_prompts[
            len(assistant_prompts) -
            len(user_prompts):
        ]

        return [
            system_prompt,
            *[a for b in zip(
                user_prompts,
                assistant_prompts,
            ) for a in b],
            last_user_prompt
        ]
