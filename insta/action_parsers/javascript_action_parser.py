from insta.utils import (
    BrowserAction,
    BrowserStatus,
    FunctionCall
)
from insta.action_parsers.base_action_parser import (
    BaseActionParser
)

from typing import List
import re


ACTION_PATTERN = re.compile(
    r"```javascript\n(?P<javascript>.*)\n```",
    re.DOTALL
)


FUNCTION_PATTERN = r"(\w+(?:\.\w+)*)\(([^)]*)\)"
ID_PATTERN = r"(id)\s*=\s*([\"\'])(\d+)\2"
BACKEND_REPLACEMENT = r"backend_node_id=\2\3\2"


def id_to_backend_node_id(
    javascript: str
) -> str:
    
    return re.sub(
       ID_PATTERN,
       BACKEND_REPLACEMENT,
        javascript
    )


def get_function_calls(
    javascript: str
) -> List[FunctionCall]:
    
    matches = re.finditer(
        FUNCTION_PATTERN,
        javascript
    )

    function_calls = []

    for match in matches:

        function_call = match.group()

        if not function_call:
            continue

        dotpath = (match.group(1) or "").strip()
        args = (match.group(2) or "").strip()

        if len(dotpath) == 0:
            continue

        args = id_to_backend_node_id(args)
        
        function_calls.append(
            FunctionCall(
                dotpath = dotpath,
                args = args
            )
        )

    return function_calls


SYSTEM_PROMPT = """You are a helpful assistant operating my web browser. I will show you the viewport of webpages rendered in markdown (to see more, you need to scroll), and I want your help to complete a web navigation task. Read the webpage, and respond with an action to interact with the page, and help me complete the task.

## Formatting The Response

Respond with actions in the following schema:

```javascript
function_name(...function_args)
```

I'm using the Node.js API for Playwright, a browser automation library, to interact with the webpage. Your actions will be executed as JavaScript function calls in a sandbox environment with the following objects:

* `browser`: A Playwright Browser object.
* `context`: A Playwright BrowserContext object.
* `page`: A Playwright Page object.

In addition, you have access to the following functions:

* `stop`: A function to stop the browsing session and return an answer.

Here are some of the available actions:

### Click Action Definition

* `page.locator("[id='$ELEMENT_ID']").click()`: Click on an element specified by `ELEMENT_ID`.

### Example Click Action

Suppose you want to click the link `[id: 5] Sales link`:

```javascript
page.locator("[id='5']").click()
```

### Hover Action Definition

* `page.locator("[id='$ELEMENT_ID']").hover()`: Hover over an element specified by `ELEMENT_ID`.

### Example Hover Action

Suppose you want to move the mouse over the image `[id: 2] Logo image`:

```javascript
page.locator("[id='2']").hover()
```

### Scroll Action Definition

* `page.mouse.wheel($DELTA_X, $DELTA_Y)`: Scroll the page by `DELTA_X` pixels to the right and `DELTA_Y` pixels down.

### Example Scroll Action

Suppose you want to scroll down the page by 300 pixels:

```javascript
page.mouse.wheel(0, 300)
```

### Fill Action Definition

* `page.locator("[id='$ELEMENT_ID']").fill("$VALUE")`: Fill an input element (including text fields, and range sliders) specified by `ELEMENT_ID` with the value `VALUE`.

### Example Fill Action (Text Input)

Suppose you want to fill the input `[id: 13] "Name..." (Enter your name text field)` with the text `John Doe`:

```javascript
page.locator("[id='13']").fill("John Doe")
```

### Example Fill Action (Range Slider)

Suppose you want to set the value of a range slider `[id: 71] "$250 (5)" (range slider min: 0 max: 50 step: 1)` to $1000:

This slider has a range of 0 to 50 with a step of 1, and the value is currently set to 5. You must translate the desired "$1000" to the correct underlying range value.

```javascript
page.locator("[id='71']").fill("20")
```

### Select Action Definition

* `page.locator("[id='$ELEMENT_ID']").selectOption("$VALUE")`: Select an option specified by `VALUE` from a select element specified by `ELEMENT_ID`.

### Example Select Action

Suppose you want to select the option `red` from the dropdown `[id: 67] "blue" (color select from: red, blue, green)`:

```javascript
page.locator("[id='67']").selectOption("red")
```

### Set Checked Action Definition

* `page.locator("[id='$ELEMENT_ID']").setChecked($CHECKED)`: Check or uncheck a checkbox specified by `ELEMENT_ID`.

### Example Set Checked Action

Suppose you want to check the checkbox `[id: 21] "I agree to the terms and conditions" (checkbox)`:

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

Suppose you want to open google search:

```javascript
page.goto("https://www.google.com")
```

### Stop Action Definition

* `stop("$ANSWER")`: Stop the browsing session and return the answer `ANSWER`.

### Example Stop Action

Suppose you want to stop and report `I'm done!`:

```javascript
stop("I'm done!")
```

Thanks for helping me perform tasks on the web, please follow the instructions carefully. Format your response with a summary of what you have done so far, a step-by-step explanation of your reasoning, and exactly one action you would like to perform. Limit your response to 500 words."""


USER_PROMPT_TEMPLATE = """Here is the current viewport rendered in markdown:

{observation}

{instruction}

Enter an action in the following schema:

```javascript
function_name(...function_args)
```"""


class JavascriptActionParser(BaseActionParser):
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

        action: str
            The response from an LLM that contains an action embedded in code,
            which will be parsed into a BrowserAction object.
        
        
        Returns:

        BrowserAction | PlaywrightStatus
            A BrowserAction object that contains a sequence of function
            calls to perform in a web browsing session, or a PlaywrightStatus
            object that indicates parsing the action failed.
        
        """
        
        match = ACTION_PATTERN.search(
            response
        )

        is_valid = (
            match is not None and 
            "javascript" in match.groupdict()
        )

        if not is_valid: return BrowserStatus.ERROR

        matched_response = match.group("javascript")
        function_calls = get_function_calls(
            matched_response
        )

        if len(function_calls) == 0:

            return BrowserStatus.ERROR
        
        playwright_action = BrowserAction(
            function_calls = function_calls,
            response = response,
            matched_response = matched_response
        )

        return playwright_action
