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


```javascript
function_name(...function_args)
```

I'm using the Node.js API for Playwright, a browser automation library, to interact with the webpage. The script will produce actions that will be executed as JavaScript function calls in a sandbox environment with the following objects:

* `browser`: A Playwright Browser object.
* `context`: A Playwright BrowserContext object.
* `page`: A Playwright Page object.

In addition, the script can access to the following functions:

* `stop`: A function to stop the browsing session and return an answer.

Here are some of the available actions:

### Click Action Definition

* `page.locator("[id='$ELEMENT_ID']").click()`: Click on an element specified by `ELEMENT_ID`.

### Example Click Action

Here is an example where the script clicks the link `[id: 5] Sales link`:

```javascript
page.locator("[id='5']").click()
```

### Hover Action Definition

* `page.locator("[id='$ELEMENT_ID']").hover()`: Hover over an element specified by `ELEMENT_ID`.

### Example Hover Action

Here is an example where the script moves the mouse over the image `[id: 2] Logo image`:

```javascript
page.locator("[id='2']").hover()
```

### Scroll Action Definition

* `page.mouse.wheel($DELTA_X, $DELTA_Y)`: Scroll the page by `DELTA_X` pixels to the right and `DELTA_Y` pixels down.

### Example Scroll Action

Here is an example where the script scrolls down the page by 300 pixels:

```javascript
page.mouse.wheel(0, 300)
```

### Fill Action Definition

* `page.locator("[id='$ELEMENT_ID']").fill("$VALUE")`: Fill an input element (including text fields, and range sliders) specified by `ELEMENT_ID` with the value `VALUE`.

### Example Fill Action (Text Input)

Here is an example where the script fills the input `[id: 13] "Name..." (Enter your name text field)` with the text `John Doe`:

```javascript
page.locator("[id='13']").fill("John Doe")
```

### Example Fill Action (Range Slider)

Here is an example where the script sets the value of a range slider `[id: 71] "$250 (5)" (range slider min: 0 max: 50 step: 1)` to $1000:

This slider has a range of 0 to 50 with a step of 1, and the value is currently set to 5. The script translated the desired "$1000" to the correct underlying range value.

```javascript
page.locator("[id='71']").fill("20")
```

### Select Action Definition

* `page.locator("[id='$ELEMENT_ID']").selectOption("$VALUE")`: Select an option specified by `VALUE` from a select element specified by `ELEMENT_ID`.

### Example Select Action

Here is an example where the script selects the option `red` from the dropdown `[id: 67] "blue" (color select from: red, blue, green)`:

```javascript
page.locator("[id='67']").selectOption("red")
```

### Set Checked Action Definition

* `page.locator("[id='$ELEMENT_ID']").setChecked($CHECKED)`: Check or uncheck a checkbox specified by `ELEMENT_ID`.

### Example Set Checked Action

Here is an example where the script checks the checkbox `[id: 21] "I agree to the terms and conditions" (checkbox)`:

```javascript
page.locator("[id='21']").setChecked(true)
```

### Go Back Action Definition

* `page.goBack()`: Navigate back to the previous page.

### Example Go Back Action

```javascript
page.goBack()
```

### Goto Action Definition

* `page.goto("$URL")`: Navigate to the URL specified by `URL`.

### Example Goto Action

Here is an example where the script opens google search:

```javascript
page.goto("https://www.google.com")
```

### Stop Action Definition

* `stop("$ANSWER")`: Stop the browsing session and return the answer `ANSWER`.

### Example Stop Action

Here is an example where the script stops and reports `I'm done!`:

```javascript
stop("I'm done!")
```

## Formatting The Response

Think step by step, and start your response with an explanation of your reasoning in 50 words. Then, provide an evaluation in the following JSON schema:

```json
{
    "task_is_feasible": float,
    "success": float,
    "on_right_track": float
}
```

Here is what each key means:

- `task_is_feasible`: What is the probability the desired task is feasible, rated from 0.0 (not possible) to 1.0 (absolutely certain)?
- `success`: What is the probability the desired task has been completed successfully, rated from 0.0 (not possible) to 1.0 (absolutely certain)?
- `on_right_track`: What is the probability the script is on the right track towards a future success, rated from 0.0 (not possible) to 1.0 (absolutely certain)?

Thanks for helping me evaluate the script, please follow the instructions carefully. Format your response with a summary of what the script has done so far, a step-by-step explanation of your reasoning, and exactly one evaluation following the JSON schema provided above."""


USER_PROMPT_TEMPLATE = """The desired task is: {instruction}

My script has taken the following actions so far:

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
```"""


class JavascriptJudgmentParser(BaseJudgmentParser):
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
