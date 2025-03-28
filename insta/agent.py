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

from typing import List, Callable
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

    def get_action(
        self, context: List[dict],
        max_history: int = 0
    ) -> BrowserAction | BrowserStatus:
        """Queries the LLM for the next action to take, given previous 
        actions and observations, up to a maximum history length.

        Arguments:

        context: List[dict]
            The context to use for generating the next action, which
            includes the system prompt, previous observations,
            and the current observation.

        max_history: int
            The maximum number of previous observations to include
            in the context, defaults to 0.

        Returns:

        PlaywrightAction | PlaywrightStatus
            The next action to take, or an error status the action
            failed to parse from the LLM output.
        
        """
        
        messages = self.get_prompts(
            context = context,
            max_history = max_history
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
        self, observation: str, instruction: str,
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

        Returns:

        PlaywrightAction | None
            The next action to take, or the NULL_ACTION if the action
            failed to parse from the LLM output.
        
        """

        self.push_observation(
            observation = observation,
            instruction = instruction
        )

        action = safe_call(
            self.get_action,
            context = self.context,
            max_history = self.config.max_history,
            catch_errors = self.config.catch_errors,
            max_errors = self.config.max_errors,
            log_errors = self.config.log_errors
        )

        if action is BrowserStatus.ERROR:

            return NULL_ACTION

        return action
    
    @property
    def system_prompt(self) -> str:

        return self.action_parser.system_prompt
    
    @property
    def user_prompt_template(self) -> str:

        return self.action_parser.user_prompt_template

    def get_user_prompt(self, observation: str, instruction: str) -> str:
        """Returns the user prompt for the latest observation from
        the environment, and the current user instruction.

        Arguments:

        observation: str
            The current webpage processed into an agent-readible format,
            such as the markdown format used with InSTA.

        instruction: str
            The instruction to provide to the agent, such as a question
            or a command to execute on the web.

        Returns:

        str
            The user prompt for the latest observation and instruction.
        
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
            instruction = instruction
        )

    def reset(self) -> None:
        """Reset the context for the LLM agent, and remove previous 
        observations and actions, which should be performed right after
        calling the environment reset method.
        
        """

        self.context = [{
            "role": "system",
            "content": self.system_prompt
        }]

    def push_observation(self, observation: str, instruction: str):
        """Pushes the latest observation and instruction to the context
        for the LLM agent, which is used for generating the next action.

        Arguments:

        observation: str
            The current webpage processed into an agent-readible format,
            such as the markdown format used with InSTA.

        instruction: str
            The instruction to provide to the agent, such as a question
            or a command to execute on the web.

        """

        user_prompt = self.get_user_prompt(
            observation = observation,
            instruction = instruction
        )

        self.context.append({
            "role": "user",
            "content": user_prompt
        })

    def push_action(self, response: str):
        """Add the specified agent response to the context, manual
        to handle cases where the agent response is not generated
        by the LLM, or is post-processed by the developer.

        Arguments:

        response: str
            The last response generated by the agent, which includes
            a chain of thought section, and code for an action.
        
        """

        self.context.append({
            "role": "assistant",
            "content": response
        })

    def pop_observation(self) -> dict | None:
        """If the final item in the context is an observation, then
        pop the observation from the context and return it.

        Returns:

        dict | None
            The last observation, removed from the context, or None if
            there is no observation at the end of the context.
        
        """

        has_last_observation = (
            len(self.context) > 0 and 
            self.context[-1]["role"] == "user"
        )

        if has_last_observation:
            
            return self.context.pop()

    def pop_action(self) -> dict | None:
        """If the final item in the context is an action, then
        pop the action from the context and return it.

        Returns:

        dict | None
            The last action, removed from the context, or None if
            there is no action at the end of the context.

        """

        has_last_action = (
            len(self.context) > 0 and
            self.context[-1]["role"] == "assistant"
        )

        if has_last_action:

            return self.context.pop()

    def pop_context(self) -> List[dict]:
        """Pop and reset the agent context, returning the previous context,
        which contains previous observations, and actions the agent has
        performed in the current browsing session.

        Returns:

        List[dict]
            The entire context, which contains previous observations, and 
            actions the agent has performed in the browser.

        """

        previous_context = self.context
        self.reset()
        return previous_context
    
    def get_context(self) -> List[dict]:
        """Returns the current context for the agent, which includes
        previous observations, and actions the agent has performed
        in the current browsing session.
        
        Returns:
        
        List[dict]
            The entire context, which contains previous observations, and 
            actions the agent has performed in the browser.
            
        """
            
        return self.context
    
    def set_context(self, context: List[dict]):
        """Replace the agent context with the specified context, which
        includes target observations, and actions the agent has performed
        in a different browsing session.
        
        Arguments:
        
        context: List[dict]
            An entire context, which contains previous observations, and 
            actions an agent has performed in the browser.
        
        """

        self.context = context

    def get_prompts(
        self, context: List[dict],
        max_history: int = 0
    ) -> List[dict]:
        """Select the `max_history` most recent observations and actions
        from the context, and return the context with the system prompt,
        and the most recent observations and actions.

        Arguments:

        context: List[dict]
            The entire context, which contains previous observations, and 
            actions the agent has performed in the browser.

        max_history: int
            The maximum number of previous observations to include
            in the context, defaults to 0.
        
        """

        system_prompt, *history, current_observation = context

        if max_history == 0:

            return [
                system_prompt,
                current_observation
            ]

        observations = [
            x for x in history 
            if x["role"] == "user"
        ]

        actions = [
            x for x in history 
            if x["role"] == "assistant"
        ]

        observations = observations[-max_history:]
        actions = actions[-max_history:]

        partial_action_obs = [
            x for pair in zip(observations, actions)
            for x in pair
        ]

        context = (
            [system_prompt] + 
            partial_action_obs + 
            [current_observation]
        )

        return context
