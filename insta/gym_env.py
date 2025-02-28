from insta.utils import (
    BrowserStatus,
    BrowserObservation,
    BrowserAction,
    EnvError,
    ServerError,
    ERROR_TO_MESSAGE
)

from insta.configs.browser_config import (
    BrowserConfig,
    DEFAULT_BROWSER_CONFIG,
)

from insta.client import (
    BrowserClient
)

from insta.observation_processors import (
    OBSERVATION_PROCESSORS
)

from insta.candidates import (
    CANDIDATES
)

from typing import Tuple, Any
from collections import namedtuple

import gymnasium


InstaEnvResetOuptut = namedtuple(
    "InstaEnvResetOuptut",
    ["observation", "info"]
)


InstaEnvStepOutput = namedtuple(
    "InstaEnvStepOutput",
    ["observation", "reward", "done", "truncated", "info"]
)


def return_reset_error(error: EnvError | ServerError) -> \
        Tuple[BrowserObservation, dict[str, Any]]:
    """Process an error that occurred during a reset in the Playwright
    environment and return the error message as the next observation.

    Arguments:

    error: EnvError | ServerError
        The error that occurred during the step in the Playwright environment,
        which has an error message to propagate to the agent, allowing
        the agent to recover and continue attempting the task.

    Returns:

    Tuple[PlaywrightObservation, float, bool, bool, dict[str, Any]]
        The next observation, the reward for the action, whether the episode
        is done, whether the environment has finished early,
        and metadata about the web browsing session.
    
    """
    
    error_message = (
        ERROR_TO_MESSAGE[error]
        if isinstance(error, EnvError)
        else error.message
    )
    
    processed_obs = BrowserObservation(
        processed_text = 
        error_message
    )

    return InstaEnvResetOuptut(
        observation = processed_obs,
        info = {}
    )


def return_step_error(error: EnvError | ServerError) -> \
        Tuple[BrowserObservation, float, bool, bool, dict[str, Any]]:
    """Process an error that occurred during a step in the Playwright
    environment and return the error message as the next observation.

    Arguments:

    error: EnvError | ServerError
        The error that occurred during the step in the Playwright environment,
        which has an error message to propagate to the agent, allowing
        the agent to recover and continue attempting the task.

    Returns:

    Tuple[PlaywrightObservation, float, bool, bool, dict[str, Any]]
        The next observation, the reward for the action, whether the episode
        is done, whether the environment has finished early,
        and metadata about the web browsing session.
    
    """
    
    error_message = (
        ERROR_TO_MESSAGE[error]
        if isinstance(error, EnvError)
        else error.message
    )
    
    processed_obs = BrowserObservation(
        processed_text = 
        error_message
    )

    truncated = (
        "Session ID not found"
        in error_message
    )

    return InstaEnvStepOutput(
        observation = processed_obs,
        reward = 0.0,
        done = truncated,
        truncated = truncated,
        info = {}
    )


class InstaEnv(gymnasium.Env):
    """Initialize a web browsing environment for training LLM agents, this
    environment provides a clean interface for LLM agents to control a browser
    and actions are taken on webpages using Playwright API function calls.

    Attributes:

    config: EnvConfig
        The configuration for the Playwright environment, refer to
        insta/configs/env_config.py for more information.

    client: PlaywrightClient
        A client that connects to a server running Playwright and
        manages web browsing sessions, actions, and observations.

    observation_processor: BaseProcessor
        A processor that converts HTML into text for the agent to read.

    action_parser: BaseActionParser
        A parser that converts text into function calls for the agent to execute.

    candidates: BaseCandidates
        A preprocessor that identifies interactive elements on the webpage.

    initialized: bool
        Whether the environment has been initialized.

    """

    def __init__(self, config: BrowserConfig = DEFAULT_BROWSER_CONFIG,
                 observation_processor: str = "markdown",
                 candidates: str = "all"):
        """Initialize a web browsing environment for training LLM agents, this
        environment provides a clean interface for LLM agents to control a browser
        and actions are taken on webpages using Playwright API function calls.

        Arguments:

        config: EnvConfig
            The configuration for the Playwright environment, refer to
            insta/configs/env_config.py for more information.
        
        observation_processor: str
            The observation processor to use for converting HTML to text,
            currently you can select from: ["markdown"].

        candidates: str
            The candidates to use for identifying interactive elements,
            currently you can select from: ["all"]

        """
        
        super(InstaEnv, self).__init__()

        self.config = config
        self.client = BrowserClient(
            config = config
        )

        self.observation_processor = OBSERVATION_PROCESSORS[
            observation_processor
        ]()

        self.candidates = CANDIDATES[
            candidates
        ]()

    def get_obs(self) -> BrowserObservation:
        """Get the current observation from the Playwright environment,
        process the observation to find interactive elements, and
        convert the HTML into an agent-readible text format.

        Returns:

        PlaywrightObservation
            An instance of PlaywrightObservation containing the processed
            text or an error message if something failed.
        
        """
        
        obs = self.client.observation()

        if obs is BrowserStatus.ERROR:

            x = ERROR_TO_MESSAGE[
                EnvError.PROCESSING_ERROR
            ]
            
            return BrowserObservation(
                processed_text = x
            )
        
        if isinstance(obs, ServerError):
            
            return BrowserObservation(
                processed_text = 
                obs.message
            )
        
        self.candidates.update(obs)
        
        return self.observation_processor.process(
            obs, restrict_viewport = self.config.restrict_viewport,
            require_visible = self.config.require_visible,
            require_frontmost = self.config.require_frontmost
        )

    def reset(self, url: str, browser_kwargs = None, context_kwargs = None
              ) -> Tuple[BrowserObservation, dict[str, Any]]:
        """Reset the Playwright environment to a new webpage and return
        the initial observation from the browser, or return an error if
        the environment fails to reset or initialize.

        Arguments:

        url: str
            The URL of the webpage to reset the environment to.

        browser_kwargs: dict
            Keyword options to use when starting the Browser instance,
            refer to the Playwright docs for more information.

        context_kwargs: dict
            Keyword options to use when starting the BrowserContext instance,
            refer to the Playwright docs for more information.

        Returns:

        Tuple[PlaywrightObservation, dict[str, Any]]
            An initial observation from the Playwright environment and 
            metadata about the state of the browsing session.
        
        """

        start_status = self.client.start(
            browser_kwargs = browser_kwargs,
            context_kwargs = context_kwargs
        )

        if start_status is BrowserStatus.ERROR:

            return return_reset_error(
                EnvError.START_ERROR
            )

        if isinstance(start_status, ServerError):

            return return_reset_error(
                start_status
            )

        goto_status = self.client.goto(url = url)

        if goto_status is BrowserStatus.ERROR:

            return return_reset_error(
                EnvError.GOTO_ERROR
            )

        if isinstance(goto_status, ServerError):

            return return_reset_error(
                goto_status
            )
        
        return InstaEnvResetOuptut(
            observation = self.get_obs(), 
            info = {}
        )

    def step(self, action: BrowserAction) -> \
            Tuple[BrowserObservation, float, bool, bool, dict[str, Any]]:
        """Take an action, and return the next observation, the reward
        for the action, whether the episode is done,  whether the environment
        has finished early, and metadata about the web browsing session.

        Arguments:

        action: PlaywrightAction
            The action to perform in the web browsing session, represented as a 
            sequence of function calls within the Playwright API.

        Returns:

        Tuple[PlaywrightObservation, float, bool, bool, dict[str, Any]]
            The next observation, the reward for the action, whether the episode
            is done, whether the environment has finished early,
            and metadata about the web browsing session.

        """
        
        is_stop_action = (
            action.function_calls[0].dotpath
            .startswith("stop")
        )

        if is_stop_action:

            return InstaEnvStepOutput(
                observation = self.get_obs(),
                reward = 0.0,
                done = True,
                truncated = False,
                info = {}
            )

        action_status = self.client.action(
            function_calls = action.function_calls
        )

        if action_status is BrowserStatus.ERROR:

            return return_step_error(
                EnvError.ACTION_FAILED_ERROR
            )

        if isinstance(action_status, ServerError):

            return return_step_error(
                action_status
            )
        
        return InstaEnvStepOutput(
            observation = self.get_obs(),
            reward = 0.0,
            done = False,
            truncated = False,
            info = {}
        )
