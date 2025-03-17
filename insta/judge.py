from insta.judgment_parsers import (
    JUDGMENT_PARSERS
)

from insta.utils import (
    safe_call,
    BrowserStatus
)

from insta.utils import (
    BrowserJudgment
)

from insta.configs.judge_config import (
    JudgeConfig,
    DEFAULT_JUDGE_CONFIG
)

from typing import List, Callable
from transformers import AutoTokenizer

import openai


NULL_JUDGMENT = BrowserJudgment(
    values = {},
    response = None,
    matched_response = None
)


class BrowserJudge(Callable):
    """Defines an LLM Judge for evaluating agents operating a web browser,
    served via the OpenAI API---local LLMs can be served using vLLM, 
    and proprietary LLMs can be accessed directly through the OpenAI API.

    Attributes: 

    config: JudgeConfig
        The configuration for the judge, which includes the tokenizer
        to use, the client to use, and the generation kwargs to use,
        refer to insta/configs/judge_config.py for more information.

    tokenizer: AutoTokenizer
        The tokenizer to use for encoding and decoding text, which is
        used for truncating observation text to a max length.

    judgment_parser: JudgmentParser
        The judgment parser for parsing responses from the LLM into a
        dictionary of scores that estimate the agent's performance,
        refer to insta/judgment_parsers/* for more information.

    llm_client: openai.OpenAI
        The OpenAI client for querying the LLM, provides a standard
        interface for connecting to a variety of LLMs, including
        local LLMs served via vLLM, GPT, and Gemini models
    
    """

    def __init__(self, config: JudgeConfig = DEFAULT_JUDGE_CONFIG,
                 judgment_parser: str = "json"):
        """Defines an LLM Judge for evaluating agents operating a web browser,
        served via the OpenAI API---local LLMs can be served using vLLM, 
        and proprietary LLMs can be accessed directly through the OpenAI API.

        Arguments:

        config: JudgeConfig
            The configuration for the judge, which includes the tokenizer
            to use, the client to use, and the generation kwargs to use,
            refer to insta/configs/judge_config.py for more information.


        judgment_parser: str
            The judgment parser for parsing responses from the LLM into a
            dictionary of scores that estimate the agent's performance,
            refer to insta/judgment_parsers/* for more information.
        
        """

        super(BrowserJudge, self).__init__()

        self.config = config

        self.judgment_parser = JUDGMENT_PARSERS[
            judgment_parser
        ]()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer
        )

        self.llm_client = openai.OpenAI(
            **self.config.client_kwargs
        )

    def get_judgment(
        self, observations: List[str], 
        actions: List[str],
        instruction: str,
        last_actions: int = 5,
        last_obs: int = 5
    ) -> BrowserJudgment | BrowserStatus:
        """Queries the LLM a judgment that estimates the agent's performance
        given the instruction, observations, and actions.

        Arguments:

        observations: List[str]
            The current webpage processed into an agent-readible format,
            such as the markdown format used with InSTA.

        actions: List[str]
            The previous actions the agent has taken in the browser,
            typically the raw LLM action output.

        instruction: str
            The instruction to provide to the agent, such as a question
            or a command to execute on the web.

        last_actions: int
            The number of actions to include in the context.

        last_obs: int
            The number of observations to include in the context.

        Returns:

        BrowserJudgment | BrowserStatus
            An estimate of the agent's performance, or an error if the
            judgment failed to parse from the LLM output.

        """
        
        messages = self.get_context(
            instruction = instruction,
            observations = observations,
            actions = actions,
            last_actions = last_actions,
            last_obs = last_obs
        )

        completion = self.llm_client.chat.completions.create(
            messages = messages,
            **self.config.generation_kwargs
        )

        return self.judgment_parser.parse_judgment(
            completion.choices[0]
            .message.content
        )

    def __call__(
        self, observations: List[str], 
        actions: List[str],
        instruction: str,
    ) -> BrowserJudgment | None:
        """Queries the LLM a judgment that estimates the agent's performance
        given the instruction, observations, and actions.

        Arguments:

        observations: List[str]
            The current webpage processed into an agent-readible format,
            such as the markdown format used with InSTA.

        actions: List[str]
            The previous actions the agent has taken in the browser,
            typically the raw LLM action output.

        instruction: str
            The instruction to provide to the agent, such as a question
            or a command to execute on the web.

        Returns:

        BrowserJudgment | None
            An estimate of the agent's performance, or NULL_JUDGMENT if the
            judgment failed to parse from the LLM output.
        
        """

        judgment = safe_call(
            self.get_judgment,
            observations = observations,
            actions = actions,
            instruction = instruction,
            last_actions = self.config.last_actions,
            last_obs = self.config.last_obs,
            catch_errors = self.config.catch_errors,
            max_errors = self.config.max_errors,
            log_errors = self.config.log_errors
        )

        if judgment is BrowserStatus.ERROR:

            return NULL_JUDGMENT

        return judgment
    
    @property
    def system_prompt(self) -> str:

        return self.judgment_parser.system_prompt
    
    @property
    def user_prompt_template(self) -> str:

        return self.judgment_parser.user_prompt_template

    def get_user_prompt(
        self, observations: List[str], 
        actions: List[str],
        instruction: str,
        last_actions: int = 5,
        last_obs: int = 5
    ) -> List[dict]:
        """Select the `last_actions` most recent actions from the context,
        and return a formatted user prompt that will be passed to an 
        LLM for estimating the agent's performance.

        Arguments:

        observations: List[str]
            The current webpage processed into an agent-readible format,
            such as the markdown format used with InSTA.

        actions: List[str]
            The previous actions the agent has taken in the browser,
            typically the raw LLM action output.

        instruction: str
            The instruction to provide to the agent, such as a question
            or a command to execute on the web.

        last_actions: int
            The number of actions to include in the context.

        last_obs: int
            The number of observations to include in the context.
        
        """

        outputs = []

        for step, (observation, action) in \
                enumerate(zip(observations, actions)):

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
                len(actions) - step - 1
            )

            if time_left < last_obs:

                outputs.append("### {} Webpage:\n\n{}".format(
                    "Previous" 
                    if time_left > 0 else 
                    "Last",
                    observation
                ))

            if time_left < last_actions:
    
                outputs.append("### {} Action:\n\n{}".format(
                    "Previous" 
                    if time_left > 0 else 
                    "Next",
                    action
                ))

        trajectory = "\n\n".join(outputs)
        
        return self.user_prompt_template.format(
            trajectory = trajectory,
            instruction = instruction
        )

    def get_context(
        self, observations: List[str], 
        actions: List[str],
        instruction: str,
        last_actions: int = 5,
        last_obs: int = 5
    ) -> List[dict]:
        """Construct a series of messages for querying an LLM via the
        OpenAI API to produce a judgment that estimates the
        performance of a web navigation agent.

        Arguments:

        observations: List[str]
            The current webpage processed into an agent-readible format,
            such as the markdown format used with InSTA.

        actions: List[str]
            The previous actions the agent has taken in the browser,
            typically the raw LLM action output.

        instruction: str
            The instruction to provide to the agent, such as a question
            or a command to execute on the web.

        last_actions: int
            The number of actions to include in the context.
        
        """

        user_prompt_str = self.get_user_prompt(
            instruction = instruction,
            observations = observations,
            actions = actions,
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
