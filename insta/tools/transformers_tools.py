from insta.tools.insta_tools import (
    InstaTool,
    InstaGradioTool
)

from typing import Tuple, Optional
from PIL import Image

import transformers


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

TOOL_OUTPUT_TYPE = "string"

TOOL_OUTPUT_TEMPLATE = """Here is your assigned session ID: `{session_id}`

You are visiting the URL: `{current_url}`

Here is the current viewport rendered in markdown:\n\n{observation}"""


class InstaTransformersTool(transformers.Tool):
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
    output_type = TOOL_OUTPUT_TYPE

    def __init__(self, *insta_args, **insta_kwargs):
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

        browser_kwargs: dict
            Keyword options to use when starting the Browser instance,
            refer to the Playwright docs for more information.

        context_kwargs: dict
            Keyword options to use when starting the BrowserContext instance,
            refer to the Playwright docs for more information.

        """
        
        super(InstaTransformersTool, self).__init__()
        
        self.insta_tool = InstaTool(*insta_args, **insta_kwargs)
    
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

        observation: str
            A markdown string representing the current webpage.

        """

        x = self.insta_tool(
            session_id = session_id,
            url = url,
            action = action
        )

        return TOOL_OUTPUT_TEMPLATE.format(
            session_id = x.session_id,
            current_url = url,
            observation = x.processed_text
        )


class InstaTransformersGradioTool(transformers.Tool):
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
    output_type = TOOL_OUTPUT_TYPE

    def __init__(self, *insta_args, **insta_kwargs):
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

        browser_kwargs: dict
            Keyword options to use when starting the Browser instance,
            refer to the Playwright docs for more information.

        context_kwargs: dict
            Keyword options to use when starting the BrowserContext instance,
            refer to the Playwright docs for more information.

        """
        
        super(InstaTransformersGradioTool, self).__init__()
        
        self.insta_tool = InstaGradioTool(*insta_args, **insta_kwargs)
    
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

        observation: str
            A markdown string representing the current webpage.

        """

        x = self.insta_tool(
            session_id = session_id,
            url = url,
            action = action
        )

        return TOOL_OUTPUT_TEMPLATE.format(
            session_id = x.session_id,
            current_url = url,
            observation = x.processed_text
        )
