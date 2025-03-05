from insta.utils import (
    BrowserJudgment,
    BrowserStatus
)
from insta.judgment_parsers.base_judgement_parser import (
    BaseJudgmentParser
)

from typing import List
import re
import json


ACTION_PATTERN = re.compile(
    r"```json\n(?P<json>.*)\n```",
    re.DOTALL
)


SYSTEM_PROMPT = """You are a helpful assistant providing feedback on a web automation script. I will show you a list of previous actions, the current webpage formatted in markdown, and the proposed next action. I want your help evaluating the proposed action, to determine if the desired task is complete, or if we are on the right track towards future completion.

## Reading The Action Schema

You will see actions in the following JSON schema:

```json
{
    "action_key": str,
    "action_kwargs": dict,
    "target_element_id": int
}
```

Here is what each key means:

- `action_key`: The action to perform.
- `action_kwargs`: Dictionary of arguments for action.
- `target_element_id`: The id of the element to perform the action on.

## Available Actions

I'm using playwright, a browser automation library, to interact with the page. I'm parsing the value assigned to `action_key` into a method call on the page object, or an element specified by the value assigned to `target_element_id`. Here is an example action:

### Click Action Definition

- `click`: Click on an element specified by `target_element_id`.

### Example Click Action

Here is an example where the script clicks the link `[id: 5] Sales link`:

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

Here is an example where the script hovers over the image `[id: 2] Company Logo image`:

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

Here is an example where the script scrolls down the page by 300 pixels:

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

Here is an example where the script fills the input `[id: 13] "Name..." (Enter your name text field)` with the text `John Doe`:

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

Here is an example where the script sets the value of a range slider `[id: 71] "$250 (5)" (range slider min: 0 max: 50 step: 1)` to $1000:

This slider has a range of 0 to 50 with a step of 1, and the value is currently set to 5. The script must translate the desired "$1000" to the correct underlying range value.

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

Here is an example where the script selects the option `red` from the dropdown `[id: 67] "blue" (color select from: red, blue, green)`:

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

Here is an example where the script checks the checkbox `[id: 21] "I agree to the terms and conditions" (checkbox)`:

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

Here is an example where the script opens google search:

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

Here is an example where the script stops and reports `I'm done!`:

```json
{
    "action_key": "stop",
    "action_kwargs": {
        "answer": "I'm done!"
    },
    "target_element_id": null
}
```

## Formatting The Response

Format your evaluation in the following JSON schema:

```json
{
    "task_is_feasible": float,
    "success": float,
    "on_right_track": float
}
```

Here is what each key means:

- `task_is_feasible`: The probability the desired task is feasible
    - range: 0.0 (not possible) to 1.0 (absolutely certain).
- `success`: The probability the desired task has been completed successfully.
    - range: 0.0 (not possible) to 1.0 (absolutely certain).
- `on_right_track`: The probability the script is on the right track towards a future success.
    - range: 0.0 (not possible) to 1.0 (absolutely certain).

Thanks for helping me with evaluation, please follow the instructions carefully. Start your response with a summary of what the script has accomplished, followed by a step-by-step explanation of your evaluation, and finally, provide your evaluation in the JSON format. Limit your response to 500 words."""


USER_PROMPT_TEMPLATE = """The desired task is: {instruction}

The script has taken the following actions so far:

{previous_actions}

The current webpage is:

{observation}

Here is the proposed next action:

{next_action}

Enter an evaluation in the following JSON schema:

```json
{{
    "task_is_feasible": float,
    "success": float,
    "on_right_track": float
}}
```

Start your response with a summary of what the script has accomplished, followed by a step-by-step explanation of your evaluation, and finally, provide your evaluation in the JSON format. Limit your response to 500 words."""


class JsonJudgmentParser(BaseJudgmentParser):
    """Implements a parser for converting text generated by an LLM into a
    judgment of whether a web browsing task has been successfully completed,
    returns a BrowserJudgment instance parsed from the response.

    Attributes:

    system_prompt: str
        Depending on the judgment representation, this system prompt
        instructs the LLM on how to generate judgments in the corresponding format,
        such as JSON-based judgments, python code, etc.

    user_prompt_template: str
        A template string that is used to generate a user prompt for the LLM,
        and had format keys for `observation` and `instruction`.

    """

    system_prompt = SYSTEM_PROMPT
    user_prompt_template = USER_PROMPT_TEMPLATE

    def parse_judgment(self, response: str) -> BrowserJudgment | BrowserStatus:
        """Parse a judgment string produced by an LLM, and return a
        BrowserJudgment object that contains a sequence of function calls
        to perform in a web browsing session.

        Arguments:

        response: str
            The response from an LLM that contains a judgment in a code block,
            which will be parsed into a BrowserJudgment object.
        
        Returns:

        BrowserJudgment | PlaywrightStatus
            The parser judgment object that contains a dictionary of parsed 
            judgment values, and the text values were parsed from.
        
        """
        
        match = ACTION_PATTERN.search(
            response
        )

        has_required_field = (
            match is not None and 
            "json" in match.groupdict()
        )

        if not has_required_field:
    
            return BrowserStatus.ERROR

        matched_response = match.group("json")
        
        try: response_dict = json.loads(matched_response)
        except json.JSONDecodeError:
            return BrowserStatus.ERROR
        
        has_required_keys = (
            "task_is_feasible" in response_dict and
            "success" in response_dict and
            "on_right_track" in response_dict
        )

        if not has_required_keys:

            return BrowserStatus.ERROR
        
        task_is_feasible = response_dict["task_is_feasible"]
        success = response_dict["success"]
        on_right_track = response_dict["on_right_track"]
        
        keys_right_type = (
            (isinstance(task_is_feasible, float) or isinstance(task_is_feasible, int)) and
            (isinstance(success, float) or isinstance(success, int)) and
            (isinstance(on_right_track, float) or isinstance(on_right_track, int))
        )

        if not keys_right_type:

            return BrowserStatus.ERROR
        
        values = {
            "task_is_feasible": float(task_is_feasible),
            "success": float(success),
            "on_right_track": float(on_right_track)
        }
        
        browser_judgment = BrowserJudgment(
            values = values,
            response = response,
            matched_response = matched_response
        )

        return browser_judgment
