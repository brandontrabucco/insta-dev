from insta.configs.browser_config import (
    BrowserConfig,
    DEFAULT_BROWSER_CONFIG
)

from insta.tools.api import (
    interact_with_browser
)

from typing import Dict, Tuple, Callable, Optional
from PIL import Image


TOOL_NAME = "insta-browser-environment"

TOOL_DESCRIPTION = "A tool for training LLM agents to control a browser using Playwright. It takes as input a javascript function call chain in the Playwright API, such as: `page.locator(\"[id='5']\").click()`, and executes them on the Page object."

TOOL_INPUTS = {
    "session_id": {
        "type": "string",
        "description": "The current browsing session ID, if any."
    },
    "url": {
        "type": "string",
        "description": "The URL to open in the browser."
    },
    "action": {
        "type": "string",
        "description": "The action to perform in the browser."
    }
}

TOOL_OUTPUTS = {
    "session_id": {
        "type": "string",
        "description": "The assigned session ID for the browsing session."
    },
    "processed_text": {
        "type": "string",
        "description": "Markdown representation of the viewport."
    },
    "screenshot": {
        "type": "image",
        "description": "Screenshot of the viewport."
    }
}


class InstaTool(Callable):
    """Defines a web browsing tool for training web navigation agents, this
    tool provides a clean interface for LLM agents to control a browser
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

    name: str
        The name of the tool.

    description: str
        A description of the tool.

    inputs: List[str]
        The input types that the tool accepts.

    outputs: List[str]
        The output types that the tool returns.

    """

    name = TOOL_NAME
    description = TOOL_DESCRIPTION
    inputs = TOOL_INPUTS
    outputs = TOOL_OUTPUTS

    def __init__(self, base_config: BrowserConfig = DEFAULT_BROWSER_CONFIG,
                 shorter_session_id: bool = True,
                 playwright_url: str = "http://localhost:{port}",
                 playwright_port: int = 3000,
                 playwright_workers: int = 8,
                 observation_processor: str = "markdown",
                 action_parser: str = "javascript",
                 candidates: str = "all",
                 browser_kwargs: dict = None,
                 context_kwargs: dict = None):
        """Initialize a web browsing tool for training LLM agents, this  tool
        provides a clean interface for LLM agents to control a browser
        and actions are taken on webpages using Playwright API function calls.

        Arguments:

        config: EnvConfig
            The configuration for the Playwright environment, refer to
            insta/configs/env_config.py for more information.
        
        observation_processor: str
            The observation processor to use for converting HTML to text,
            currently you can select from: ["markdown"].

        action_parser: str
            The action parser to use for converting text to function calls,
            currently you can select from: ["javascript", "json"].

        candidates: str
            The candidates to use for identifying interactive elements,
            currently you can select from: ["all"]

        browser_kwargs: dict
            Keyword options to use when starting the Browser instance,
            refer to the Playwright docs for more information.

        context_kwargs: dict
            Keyword options to use when starting the BrowserContext instance,
            refer to the Playwright docs for more information.

        """
        
        super(InstaTool, self).__init__()

        self.base_config = base_config
        self.shorter_session_id = shorter_session_id
        self.active_sessions: Dict[str, any] = {}

        self.playwright_url = playwright_url
        self.playwright_port = playwright_port
        self.playwright_workers = playwright_workers

        self.observation_processor = observation_processor
        self.action_parser = action_parser
        self.candidates = candidates

        self.browser_kwargs = browser_kwargs
        self.context_kwargs = context_kwargs
    
    def __call__(
        self, session_id: Optional[str] = None,
        url: Optional[str] = None,
        action: Optional[str] = None,
    ) -> Tuple[str, str, Image.Image]:
        """Take an action in a web browsing environment, and return the next
        observation, represented as a markdown string.

        Arguments:

        session_id : str
            The current browsing session ID, if any.

        url : str
            The URL to open in the browser.

        action : str
            The action to perform via a snippet of Javascript code, or JSON
            targetting a module in the Playwright Node.js API.
        
        Returns:

        Tuple[str, str, Image.Image]
            A tuple containing the assigned session ID, the processed webpage,
            and the screenshot of the webpage.

        """
        
        return interact_with_browser(
            session_id = session_id, url = url, action = action,
            active_sessions = self.active_sessions,
            shorter_session_id = self.shorter_session_id,
            base_config = self.base_config,
            playwright_url = self.playwright_url,
            playwright_port = self.playwright_port,
            playwright_workers = self.playwright_workers,
            observation_processor = self.observation_processor,
            action_parser = self.action_parser,
            candidates = self.candidates,
            browser_kwargs = self.browser_kwargs,
            context_kwargs = self.context_kwargs
        )
