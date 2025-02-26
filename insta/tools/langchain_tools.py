from pydantic.v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
)

from insta.tools.insta_tools import (
    InstaTool,
    InstaGradioTool
)

from typing import Optional, Type


TOOL_DESCRIPTION = """A LangChain tool that uses Playwright to interact with web pages."""
TOOL_OUTPUT_TEMPLATE = """Here is your assigned session ID: `{session_id}`

You are visiting the URL: `{current_url}`

Here is the current viewport rendered in markdown:\n\n{observation}"""


class InstaToolInput(BaseModel):

    session_id: str = Field(description = "the assigned session ID for the current browsing session, if any")
    url: str = Field(description = "the URL to open in the browser")
    action: str = Field(description = "the action to perform via a snippet of Javascript code, or JSON targetting a module in the Playwright Node.js API")



class InstaLangchainTool(BaseTool):
    """Defines a web browsing tool for training web navigation agents, this
    tool provides a clean interface for LLM agents to control a browser
    and actions are taken on webpages using Playwright API function calls.

    """

    name: str = "WebBrowsingTool"
    description: str = TOOL_DESCRIPTION
    args_schema: Type[BaseModel] = InstaToolInput
    return_direct: bool = True
    insta_tool: InstaTool = None

    def __init__(self, *langchain_args, insta_args = (), insta_kwargs = {}, **langchain_kwargs):
        """Initializes a web browsing tool for training web navigation agents, this
        tool provides a clean interface for LLM agents to control a browser
        and actions are taken on webpages using Playwright API function calls.

        """

        super(InstaLangchainTool, self).__init__(*langchain_args, **langchain_kwargs)

        self.insta_tool = InstaTool(*insta_args, **insta_kwargs)

    def _run(
        self, session_id: Optional[str] = None,
        url: Optional[str] = None,
        action: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
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


class InstaLangchainGradioTool(BaseTool):
    """Defines a web browsing tool for training web navigation agents, this
    tool provides a clean interface for LLM agents to control a browser
    and actions are taken on webpages using Playwright API function calls.

    """

    name: str = "WebBrowsingTool"
    description: str = TOOL_DESCRIPTION
    args_schema: Type[BaseModel] = InstaToolInput
    return_direct: bool = True
    insta_tool: InstaGradioTool = None

    def __init__(self, *langchain_args, insta_args = (), insta_kwargs = {}, **langchain_kwargs):
        """Initializes a web browsing tool for training web navigation agents, this
        tool provides a clean interface for LLM agents to control a browser
        and actions are taken on webpages using Playwright API function calls.

        """

        super(InstaLangchainGradioTool, self).__init__(*langchain_args, **langchain_kwargs)

        self.insta_tool = InstaGradioTool(*insta_args, **insta_kwargs)

    def _run(
        self, session_id: Optional[str] = None,
        url: Optional[str] = None,
        action: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
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
