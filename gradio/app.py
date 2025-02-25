from typing import Tuple
from PIL import Image

from insta.configs.browser_config import (
    get_browser_config
)

from insta.tools.insta_tool import (
    InstaTool
)

import gradio as gr


# configure the browser and observations
BASE_CONFIG = get_browser_config(
    restrict_viewport = (0, 0, 1920, 1080),
    screen_width = 1920,
    screen_height = 1080,
)


# set True if running demo locally
SHORTER_SESSION_ID = False


# configure the Playwright backend
PLAYWRIGHT_URL = "http://localhost:{port}"
PLAYWRIGHT_PORT = 3000
PLAYWRIGHT_WORKERS = 8


INSTA_TOOL = InstaTool(
    base_config = BASE_CONFIG,
    shorter_session_id = SHORTER_SESSION_ID,
    playwright_url = PLAYWRIGHT_URL,
    playwright_port = PLAYWRIGHT_PORT,
    playwright_workers = PLAYWRIGHT_WORKERS,
    observation_processor = "markdown",
    action_parser = "javascript",
    candidates = "all",
    browser_kwargs = None,
    context_kwargs = None
)


def gradio_tool_call(
    session_id: str = None, 
    url: str = None,
    action: str = None
) -> Tuple[str, str, Image.Image]:
    """Entry point for the Gradio application, accepts the current browsing 
    session ID, the URL to open, and the next action to perform via 
    a snippet of Javascript code to modules in the Playwright Node.js API.

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

    return INSTA_TOOL(
        session_id = session_id,
        url = url,
        action = action
    )


session_id_textbox = gr.Textbox(
    label = "Session ID (copy the assigned value)",
    placeholder = "Copy the Session ID ..."
)

url_textbox = gr.Textbox(
    label = "URL",
    placeholder = "Enter a URL ..."
)

javascript_textbox = gr.Textbox(    
    label = "Javascript",
    placeholder = "Enter an action in javascript ..."
)


outout_session_id_textbox = gr.Textbox(
    label = "Session ID",
    placeholder = "Assigned Session ID ...",
    show_copy_button = True
)

processed_webpage_textbox = gr.Textbox(
    label = "Processed Webpage",
    placeholder = "Processed Webpage ..."
)

screenshot_image = gr.Image(
    label = "Screenshot"
)


gradio_app = gr.Interface(
    gradio_tool_call,
    inputs = [
        session_id_textbox,
        url_textbox,
        javascript_textbox
    ],
    outputs = [
        outout_session_id_textbox,
        processed_webpage_textbox,
        screenshot_image
    ],
    title = "InSTA Browser Environment",
)


if __name__ == "__main__":

    gradio_app.launch()
