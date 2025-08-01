from insta.judge_prompts import (
    JUDGE_PROMPTS
)

from insta.utils import (
    safe_call,
    BrowserStatus
)

from insta.configs.judge_config import (
    JudgeConfig,
    DEFAULT_JUDGE_CONFIG,
    BrowserJudgment
)

from typing import List, Callable
from transformers import AutoTokenizer

from vllm import (
    LLM, SamplingParams
)

import openai


NULL_JUDGMENT = BrowserJudgment(
    success = None,
    efficiency = None,
    self_correction = None,
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

    judge_prompt: JudgmentParser
        The judgment parser for parsing responses from the LLM into a
        dictionary of scores that estimate the agent's performance,
        refer to insta/judge_prompts/* for more information.

    llm_client: openai.OpenAI
        The OpenAI client for querying the LLM, provides a standard
        interface for connecting to a variety of LLMs, including
        local LLMs served via vLLM, GPT, and Gemini models
    
    """

    def __init__(self, config: JudgeConfig = DEFAULT_JUDGE_CONFIG):
        """Defines an LLM Judge for evaluating agents operating a web browser,
        served via the OpenAI API---local LLMs can be served using vLLM, 
        and proprietary LLMs can be accessed directly through the OpenAI API.

        Arguments:

        config: JudgeConfig
            The configuration for the judge, which includes the tokenizer
            to use, the client to use, and the generation kwargs to use,
            refer to insta/configs/judge_config.py for more information.

        """

        super(BrowserJudge, self).__init__()

        self.config = config

        self.judge_prompt = JUDGE_PROMPTS[
            self.config.judge_prompt
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
        
        messages = self.get_prompts(
            instruction = instruction,
            observations = observations,
            actions = actions,
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

        return self.judge_prompt.parse_judgment(
            response = response
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

        return self.judge_prompt.system_prompt
    
    @property
    def user_prompt_template(self) -> str:

        return self.judge_prompt.user_prompt_template

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

                outputs.append("## {} Webpage:\n\n{}".format(
                    "Previous" 
                    if time_left > 0 else 
                    "Last",
                    observation
                ))

            if time_left < last_actions:
    
                outputs.append("## {} Action:\n\n{}".format(
                    "Previous" 
                    if time_left > 0 else 
                    "Next",
                    action
                ))

        summary = "\n\n".join(outputs)
        
        return self.user_prompt_template.format(
            summary = summary,
            instruction = instruction
        )

    def get_prompts(
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
