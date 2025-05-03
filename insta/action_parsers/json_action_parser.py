from insta.action_parsers.action_parser import (
    BaseActionParser
)

from insta.utils import (
    BrowserStatus
)

from insta.configs.browser_config import (
    FunctionCall
)

from insta.configs.agent_config import (
    BrowserAction
)

from typing import List
import re
import json


ACTION_PATTERN = re.compile(
    r"```json\n(?P<json>.*)\n```",
    re.DOTALL
)


def get_function_calls(
    action_dict: dict
) -> List[FunctionCall]:
    
    function_calls = []

    target_element_id = action_dict.get(
        "target_element_id"
    )

    action_key = action_dict.get(
        "action_key"
    )

    action_kwargs = action_dict.get(
        "action_kwargs"
    )

    actions_with_target = [
        "click", "hover", "fill",
        "select_option", "set_checked"
    ]
    
    if target_element_id is not None \
            and action_key in actions_with_target:
        
        function_calls.append(
            FunctionCall(
                dotpath = "page.locator",
                args = "\"[backend_node_id='{}']\"".format(
                    target_element_id
                )
            )
        )
        
    if action_key == "click" and "x" in action_kwargs and "y" in action_kwargs:

        function_calls.append(
            FunctionCall(
                dotpath = "page.mouse.click",
                args = "{},{}".format(
                    action_kwargs.get("x", 0),
                    action_kwargs.get("y", 0)
                )
            )
        )
        
    elif action_key == "click" and target_element_id is not None:

        function_calls.append(
            FunctionCall(
                dotpath = "click",
                args = "{ force: true }"
            )
        )
        
    elif action_key == "hover" and target_element_id is not None:

        function_calls.append(
            FunctionCall(
                dotpath = "hover",
                args = ""
            )
        )

    elif action_key == "scroll":

        function_calls.append(
            FunctionCall(
                dotpath = "page.mouse.wheel",
                args = "{},{}".format(
                    action_kwargs.get("delta_x", 0),
                    action_kwargs.get("delta_y", 0)
                )
            )
        )

    elif action_key == "fill" and target_element_id is not None:

        function_calls.append(
            FunctionCall(
                dotpath = "fill",
                args = "'{}'".format(
                    action_kwargs.get("value")
                )
            )
        )

    elif action_key == "select_option" and target_element_id is not None:
        
        function_calls.append(
            FunctionCall(
                dotpath = "selectOption",
                args = "'{}'".format(
                    action_kwargs.get("label")
                )
            )
        )

    elif action_key == "set_checked" and target_element_id is not None:

        function_calls.append(
            FunctionCall(
                dotpath = "setChecked",
                args = "{}".format(
                    "true" if action_kwargs.get("checked", False) else "false"
                )
            )
        )

    elif action_key == "go_back":

        function_calls.append(
            FunctionCall(
                dotpath = "page.goBack",
                args = ""
            )
        )

    elif action_key == "go_forward":

        function_calls.append(
            FunctionCall(
                dotpath = "page.goForward",
                args = ""
            )
        )

    elif action_key == "goto":

        function_calls.append(
            FunctionCall(
                dotpath = "page.goto",
                args = "'{}'".format(
                    action_kwargs.get("url")
                )
            )
        )

    elif action_key == "stop":

        function_calls.append(
            FunctionCall(
                dotpath = "stop",
                args = "'{}'".format(
                    action_kwargs.get("answer")
                )
            )
        )

    return function_calls


SYSTEM_PROMPT = """You are a helpful assistant operating my web browser. I will show you the viewport of webpages rendered in markdown (to see more, you need to scroll), and I want your help to complete a web navigation task. Read the webpage, and respond with an action to interact with the page, and help me complete the task.

## Formatting The Response

Respond with actions in the following JSON schema:

```json
{
    "action_key": str,
    "action_kwargs": dict,
    "target_element_id": int | null
}
```

Here is what each key means:

- `action_key`: The action to perform.
- `action_kwargs`: Named arguments for the action.
- `target_element_id`: The id of the element to perform the action on.

## Available Actions

I'm using playwright, a browser automation library, to interact with the page. I'm parsing the value assigned to `action_key` into a method call on the page object, or an element object specified by the value assigned to `target_element_id`. Here are the available actions:

### Click Action Definition

- `click`: Click on an element specified by `target_element_id`.

### Example Click Action

Suppose you want to click the link `[id: 5] Sales link`:

```json
{
    "action_key": "click",
    "action_kwargs": {},
    "target_element_id": 5
}
```

### Hover Action Definition

- `hover`: Hover over an element specified by `target_element_id`

### Example Hover Action

Suppose you want to hover over the image `[id: 2] Company Logo image`:

```json
{
    "action_key": "hover",
    "action_kwargs": {},
    "target_element_id": 2
}
```

### Scroll Action Definition

- `scroll`: Scroll the page by `delta_x` pixels to the right and `delta_y` pixels down.
    - `delta_x`: The number of pixels to scroll to the right.
    - `delta_y`: The number of pixels to scroll down.

### Example Scroll Action

Suppose you want to scroll down the page by 300 pixels:

```json
{
    "action_key": "scroll",
    "action_kwargs": {
        "delta_x": 0,
        "delta_y": 300
    },
    "target_element_id": null
}
```

### Fill Action Definition

- `fill`: Fill an input element specified by `target_element_id` with text.
    - `value`: The text value to fill into the element.

### Example Fill Action (Text Input)

Suppose you want to fill the input `[id: 13] "Name..." (Enter your name text input)` with the text `John Doe`:

```json
{
    "action_key": "fill",
    "action_kwargs": {
        "value": "John Doe"
    },
    "target_element_id": 13
}
```

### Example Fill Action (Range Slider)

Suppose you want to set the value of a range slider `[id: 71] "$250 (5)" (range slider min: 0 max: 50 step: 1)` to $1000:

This slider has a range of 0 to 50 with a step of 1, and the value is currently set to 5. You must translate the desired "$1000" to the correct underlying range value.

```json
{
    "action_key": "fill",
    "action_kwargs": {
        "value": "20"
    },
    "target_element_id": 71
}
```

### Select Action Definition

- `select`: Select from a dropdown element specified by `target_element_id`.
    - `label`: The option name to select in the element.

### Example Select Action

Suppose you want to select the option `red` from the dropdown `[id: 67] "blue" (color select from: red, blue, green)`:

```json
{
    "action_key": "select_option",
    "action_kwargs": {
        "label": "red"
    },
    "target_element_id": 67
}
```

### Set Checked Action Definition

- `set_checked`: Check or uncheck a checkbox specified by `target_element_id`.
    - `checked`: Boolean value to check or uncheck the checkbox.

### Example Set Checked Action

Suppose you want to check the checkbox `[id: 21] "I agree to the terms and conditions" (checkbox)`:

```json
{
    "action_key": "set_checked",
    "action_kwargs": {
        "checked": true
    },
    "target_element_id": 21
}
```

### Go Back Action Definition

- `go_back`: Go back to the previous page (`target_element_id` must be null).

### Example Go Back Action

```json
{
    "action_key": "go_back",
    "action_kwargs": {},
    "target_element_id": null
}
```

### Goto Action Definition

- `goto`: Navigate to a new page (`target_element_id` must be null).
    - `url`: The URL of the page to navigate to.

### Example Goto Action

Suppose you want to open google search:

```json
{
    "action_key": "goto",
    "action_kwargs": {
        "url": "https://www.google.com"
    },
    "target_element_id": null
}
```

### Stop Action Definition

- `stop`: Stop the browser when the task is complete, or the answer is known.
    - `answer`: Optional answer if I requested one.

### Example Stop Action

```json
{
    "action_key": "stop",
    "action_kwargs": {
        "answer": "I'm done!"
    },
    "target_element_id": null
}
```

Thanks for helping me perform tasks on the web, please follow the instructions carefully. Start your response with a summary of what you have accomplished so far, followed by a step-by-step explanation of your plan and intended action, and finally, provide your action in the JSON format. Respond in 200 words."""


USER_PROMPT_TEMPLATE = """You are currently viewing {current_url}. Here is the viewport rendered in markdown:

{observation}

{instruction}

Enter an action in the following JSON schema:

```json
{{
    "action_key": str,
    "action_kwargs": dict,
    "target_element_id": int | null,
}}
```

Start your response with a summary of what you have accomplished so far, followed by a step-by-step explanation of your plan and intended action, and finally, provide your action in the JSON format. Respond in 200 words."""


class JsonActionParser(BaseActionParser):
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

    system_prompt = SYSTEM_PROMPT
    user_prompt_template = USER_PROMPT_TEMPLATE

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
        
        match = ACTION_PATTERN.search(response)

        is_valid = (
            match is not None and 
            "json" in match.groupdict()
        )

        if not is_valid:
    
            raise ValueError(
                "Failed to parse action"
            )

        matched_response = match.group("json")

        response_dict = json.loads(
            matched_response
        )

        function_calls = get_function_calls(
            response_dict
        )

        if len(function_calls) == 0:
    
            raise ValueError(
                "Failed to parse action"
            )
        
        playwright_action = BrowserAction(
            function_calls = function_calls,
            response = response,
            matched_response = matched_response
        )

        return playwright_action
