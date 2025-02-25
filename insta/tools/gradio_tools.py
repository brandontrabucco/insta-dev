from typing import Tuple, Callable, Optional
from PIL import Image

from insta.tools.insta_tool import (
    TOOL_NAME,
    TOOL_DESCRIPTION,
    TOOL_INPUTS,
    TOOL_OUTPUTS
)

from insta.tools.api import (
    InstaToolOutput
)

from gradio_client import Client


GRADIO_SERVER = "http://insta.btrabuc.co:7860"


class InstaGradioTool(Callable):
    """Defines a web browsing tool for training web navigation agents, this
    tool provides a clean interface for LLM agents to control a browser
    and actions are taken on webpages using Playwright API function calls.

    """

    name = TOOL_NAME
    description = TOOL_DESCRIPTION
    inputs = TOOL_INPUTS
    outputs = TOOL_OUTPUTS

    def __init__(self, *insta_args, src = GRADIO_SERVER, **insta_kwargs):
        """Initializes a web browsing tool for training web navigation agents, this
        tool provides a clean interface for LLM agents to control a browser
        and actions are taken on webpages using Playwright API function calls.

        """
        
        super(InstaGradioTool, self).__init__()

        self.gradio_client = Client(
            *insta_args, src = src, **insta_kwargs
        )

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

        session_id, processed_text, screenshot = self.gradio_client.predict(
            session_id = session_id, url = url, action = action
        )

        return InstaToolOutput(
            session_id = session_id,
            processed_text = processed_text,
            screenshot = screenshot
        )