from insta.action_parsers import (
    ACTION_PARSERS
)

from insta.utils import (
    safe_call,
    BrowserStatus
)

from insta.configs.agent_config import (
    AgentConfig,
    DEFAULT_AGENT_CONFIG,
    BrowserAction
)

from typing import List, Callable, Tuple, Dict
from transformers import AutoTokenizer

import openai


NULL_ACTION = BrowserAction(
    function_calls = [],
    response = None,
    matched_response = None
)


class BrowserAgent(Callable):
    """Defines an LLM Agent for interacting with a web browsing session,
    served via the OpenAI API---local LLMs can be served using vLLM, 
    and proprietary LLMs can be accessed directly through the OpenAI API.

    Attributes: 

    config: AgentConfig
        The configuration for the agent, which includes the tokenizer
        to use, the client to use, and the generation kwargs to use,
        refer to insta/configs/agent_config.py for more information.

    tokenizer: AutoTokenizer
        The tokenizer to use for encoding and decoding text, which is
        used for truncating observation text to a max length.

    action_parser: ActionParser
        The action parser for parsing the output of the LLM into a
        sequence of function calls to the Playwright API,
        refer to insta/action_parsers.py for more information.

    llm_client: openai.OpenAI
        The OpenAI client for querying the LLM, provides a standard
        interface for connecting to a variety of LLMs, including
        local LLMs served via vLLM, GPT, and Gemini models
    
    """

    observations: List[str]
    instructions: List[str]
    urls: List[str]
    actions: List[str]

    def __init__(self, config: AgentConfig = DEFAULT_AGENT_CONFIG):
        """Defines an LLM Agent for interacting with a web browsing session,
        served via the OpenAI API---local LLMs can be served using vLLM, 
        and proprietary LLMs can be accessed directly through the OpenAI API.

        Arguments:

        config: AgentConfig
            The configuration for the agent, which includes the tokenizer
            to use, the client to use, and the generation kwargs to use,
            refer to insta/configs/agent_config.py for more information.
        
        """

        super(BrowserAgent, self).__init__()

        self.config = config

        self.action_parser = ACTION_PARSERS[
            self.config.action_parser
        ]()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer
        )

        self.llm_client = openai.OpenAI(
            **self.config.client_kwargs
        )

        self.reset()

    def reset(self) -> None:
        """Reset the context for the LLM agent, and remove previous 
        observations and actions, which should be performed right after
        calling the environment reset method.
        
        """

        self.observations = []
        self.instructions = []
        self.urls = []
        self.actions = []

    def get_action(
        self, observations: List[str], 
        instructions: List[str],
        urls: List[str],
        actions: List[str],
        last_actions: int = 0
    ) -> BrowserAction | BrowserStatus:
        """Queries the LLM for the next action to take, given previous 
        actions and observations, up to a maximum history length.

        Arguments:

        observations: List[str]
            The current webpage processed into an agent-readible format,
            such as the markdown format used with InSTA.

        instructions: List[str]
            Instructions provided to the agent, such as questions
            or commands to execute on the web.

        urls: List[str]
            The current URL of the webpage, which is used for tracking
            the current state of the browsing session.

        actions: List[str]
            The previous actions the agent has taken in the browser,
            typically the raw LLM action output.

        last_actions: int
            The number of previous actions to include in the context.

        Returns:

        PlaywrightAction | PlaywrightStatus
            The next action to take, or an error status the action
            failed to parse from the LLM output.
        
        """
        
        messages = self.get_prompts(
            observations = observations,
            instructions = instructions,
            urls = urls,
            actions = actions,
            last_actions = last_actions
        )

        completion = self.llm_client.chat.completions.create(
            messages = messages,
            **self.config.generation_kwargs
        )

        return self.action_parser.parse_action(
            completion.choices[0]
            .message.content
        )
    
    def __call__(
        self, observation: str,
        instruction: str,
        current_url: str
    ) -> BrowserAction | None:
        """Queries the LLM for the next action to take, given previous 
        actions and observations, up to a maximum history length.

        Arguments:
        
        observation: str
            The current webpage processed into an agent-readible format,
            such as the markdown format used with InSTA.

        instruction: str
            The instruction to provide to the agent, such as a question
            or a command to execute on the web.

        current_url: str
            The current URL of the webpage, which is used for tracking
            the current state of the browsing session.

        Returns:

        PlaywrightAction | None
            The next action to take, or the NULL_ACTION if the action
            failed to parse from the LLM output.
        
        """

        self.push_observation(
            observation = observation,
            instruction = instruction,
            current_url = current_url
        )

        action = safe_call(
            self.get_action,
            observations = self.observations,
            instructions = self.instructions,
            urls = self.urls,
            actions = self.actions,
            last_actions = self.config.last_actions,
            catch_errors = self.config.catch_errors,
            max_errors = self.config.max_errors,
            log_errors = self.config.log_errors
        )

        if action is BrowserStatus.ERROR:

            return NULL_ACTION

        return action

    def push_observation(
        self, observation: str,
        instruction: str,
        current_url: str
    ):
        """Pushes the latest observation and instruction to the context
        for the LLM agent, which is used for generating the next action.

        Arguments:

        observation: str
            The current webpage processed into an agent-readible format,
            such as the markdown format used with InSTA.

        instruction: str
            The instruction to provide to the agent, such as a question
            or a command to execute on the web.

        current_url: str
            The current URL of the webpage, which is used for tracking
            the current state of the browsing session.

        """
        
        self.observations.append(observation)
        self.instructions.append(instruction)
        self.urls.append(current_url)

    def push_action(self, response: str):
        """Add the specified agent response to the context, manual
        to handle cases where the agent response is not generated
        by the LLM, or is post-processed by the developer.

        Arguments:

        response: str
            The last response generated by the agent, which includes
            a chain of thought, and an action.
        
        """

        self.actions.append(response)

    def pop_observation(self) -> Tuple[str, str, str] | None:
        """If the final item in the context is an observation, then
        pop the observation from the context and return it.

        Returns:

        Tuple[str, str, str] | None
            The last observation, removed from the context, or None if
            there is no observation at the end of the context.
        
        """

        has_last_observation = (
            len(self.observations) > 0 and 
            len(self.instructions) == len(self.observations) and
            len(self.urls) == len(self.observations) and
            len(self.actions) == (len(self.observations) - 1)
        )

        if has_last_observation:

            observation = self.observations.pop()
            instruction = self.instructions.pop()
            current_url = self.urls.pop()
            
            return (
                observation,
                instruction,
                current_url
            )

    def pop_action(self) -> str | None:
        """If the final item in the context is an action, then
        pop the action from the context and return it.

        Returns:

        action: str | None
            The last action, removed from the context, or None if
            there is no action at the end of the context.

        """

        has_last_action = (
            len(self.actions) > 0 and
            len(self.actions) == len(self.observations)
        )

        if has_last_action:

            return self.actions.pop()
    
    def get_context(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Returns the current context for the proposer, which includes
        previous observations, and actions the agent has performed
        in the current browsing session.
        
        Returns:
        
        Tuple[List[str], List[str], List[str], List[str]]
            The entire context, which contains previous observations, and 
            actions the agent has performed in the browser.
            
        """
            
        return (
            self.observations,
            self.instructions,
            self.urls,
            self.actions
        )
    
    def set_context(self, context: Tuple[List[str], List[str], List[str], List[str]]):
        """Replace the proposer context with the specified context, which
        includes target observations, and actions the agent has performed
        in a different browsing session.
        
        Arguments:
        
        Tuple[List[str], List[str], List[str], List[str]]
            An entire context, which contains previous observations, and 
            actions an agent has performed in the browser.
        
        """

        has_valid_context = (
            len(context) == 4 and
            all([isinstance(x, list) for x in context]) 
        )

        if not has_valid_context:

            raise ValueError(
                "Context must have the following type: "
                "Tuple[List[str], List[str], List[str], List[str]]"
            )
        
        self.observations = context[0]
        self.instructions = context[1]
        self.urls = context[2]
        self.actions = context[3]
    
    @property
    def system_prompt(self) -> str:

        return self.action_parser.system_prompt
    
    @property
    def user_prompt_template(self) -> str:

        return self.action_parser.user_prompt_template

    def get_single_user_prompt(
        self, observation: str,
        instruction: str,
        current_url: str
    ) -> str:
        """Returns the user prompt for the latest observation from
        the environment, and the current user instruction.

        Arguments:

        observation: str
            The current webpage processed into an agent-readible format,
            such as the markdown format used with InSTA.

        instruction: str
            The instruction to provide to the agent, such as a question
            or a command to execute on the web.

        current_url: str
            The current URL of the webpage, which is used for tracking
            the current state of the browsing session.

        Returns:

        user_prompt_str: str
            Prompt for the most recent step.
        
        """

        observation = self.tokenizer.encode(
            observation,
            max_length = self.config.max_obs_tokens,
            truncation = True
        )

        observation = self.tokenizer.decode(
            observation,
            skip_special_tokens = True
        )

        return self.user_prompt_template.format(
            observation = observation,
            instruction = instruction,
            current_url = current_url
        )

    def get_user_prompts(
        self, observations: List[str], 
        instructions: List[str],
        urls: List[str],
        last_actions: int = 5
    ) -> List[dict]:
        """Builds the user prompt for querying the LLM to propose a task,
        and selects the N most recent trajectories, and includes the 
        last M actions, observations, and judgments.

        Arguments:

        observations: List[str]
            The current webpage processed into an agent-readible format,
            such as the markdown format used with InSTA.

        instructions: List[str]
            Instructions provided to the agent, such as questions
            or commands to execute on the web.

        urls: List[str]
            The current URL of the webpage, which is used for tracking
            the current state of the browsing session.

        last_actions: int
            The number of previous actions to include in the context.

        Returns:

        user_prompts: List[dict]
            User prompts for each step of the trajectory.
        
        """

        user_prompts = []

        for trajectory_step, (
            observation,
            instruction,
            current_url
        ) in enumerate(zip(
            observations,
            instructions,
            urls
        )):

            time_left = (
                len(observations) - 
                trajectory_step - 1
            )

            include_step = (
                time_left < last_actions
            )

            if include_step:

                user_prompt_str = self.get_single_user_prompt(
                    observation = observation,
                    instruction = instruction,
                    current_url = current_url
                )

                user_prompts.append({
                    "role": "user",
                    "content": user_prompt_str
                })

        return user_prompts

    def get_prompts(
        self, observations: List[str], 
        instructions: List[str],
        urls: List[str],
        actions: List[str],
        last_actions: int = 5
    ) -> List[dict]:
        """Builds the user prompt for querying the LLM to propose a task,
        and selects the N most recent trajectories, and includes the 
        last M actions, observations, and judgments.

        Arguments:

        observations: List[str]
            The current webpage processed into an agent-readible format,
            such as the markdown format used with InSTA.

        instructions: List[str]
            Instructions provided to the agent, such as questions
            or commands to execute on the web.

        urls: List[str]
            The current URL of the webpage, which is used for tracking
            the current state of the browsing session.

        actions: List[str]
            The previous actions the agent has taken in the browser,
            typically the raw LLM action output.

        last_actions: int
            The number of previous actions to include in the context.

        Returns:

        user_prompts: List[dict]
            User prompts for each step of the trajectory.
        
        """

        valid_arguments = (
            len(observations) == (len(actions) + 1) and 
            len(observations) == len(urls) and 
            len(observations) == len(instructions)
        )

        if not valid_arguments:

            raise ValueError(
                "Invalid agent context."
            )

        system_prompt = {
            "role": "system",
            "content": self.system_prompt
        }

        *user_prompts, last_user_prompt = self.get_user_prompts(
            observations = observations,
            instructions = instructions,
            urls = urls,
            last_actions = last_actions
        )

        assistant_prompts = [{
            "role": "assistant",
            "content": action
        } for action in actions]

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