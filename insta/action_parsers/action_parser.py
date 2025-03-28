from insta.utils import (
    BrowserStatus
)

from insta.configs.agent_config import (
    BrowserAction
)

import abc


class BaseActionParser(abc.ABC):
    """Implements a parser for converting text generated by an LLM into a
    sequence of function calls to the Playwright API, represented as a
    BrowserAction that contains FunctionCall objects.

    Attributes:

    system_prompt: str
        Depending on the kind of action representation, this system prompt
        instructs the LLM on how to generate actions in the corresponding format,
        such as JSON-based actions, JavaScript code, etc.

    user_prompt_template: str
        A template string that is used to generate a user prompt for the LLM,
        and had format keys for `observation` and `instruction`.

    """

    system_prompt: str
    user_prompt_template: str

    @abc.abstractmethod
    def parse_action(self, response: str) -> BrowserAction | BrowserStatus:
        """Parse an action string produced by an LLM, and return a
        BrowserAction object that contains a sequence of function calls
        to perform in a web browsing session.

        Arguments:

        response: str
            The response from an LLM that contains an action embedded in code,
            which will be parsed into a BrowserAction object.
        
        
        Returns:

        BrowserAction | PlaywrightStatus
            A BrowserAction object that contains a sequence of function
            calls to perform in a web browsing session, or a PlaywrightStatus
            object that indicates parsing the action failed.
        
        """

        return NotImplemented
