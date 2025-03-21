from insta.utils import (
    BrowserStatus
)

from insta.configs.task_proposer_config import (
    BrowserTaskProposal
)

from insta.task_parsers.base_task_parser import (
    BaseTaskParser
)

from typing import List
import re
import json


TASK_PATTERN = re.compile(
    r"```json\n(?P<json>.*)\n```",
    re.DOTALL
)


SYSTEM_PROMPT = """You are a helpful assistant directing a web automation script. I will show you previous runs of the script, including previous tasks, webpages, actions, and performance reviews, formatted in markdown. I want your help assigning new tasks to the script.

## Understanding The Action Format

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

Here is an example where the script fills the input `[id: 13] "Name..." (Enter your name text input)` with the text `John Doe`:

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

## Understanding The Review Format

You will see performance reviews in the following JSON schema:

```json
{
    "task_is_feasible": float,
    "is_blocked": float,
    "success": float,
    "future_success": float,
    "reasoning_is_correct": float
}
```

Here is what each key means:

- `task_is_feasible`: The probability the desired task is feasible on this website.
    - range: 0.0 (not possible) to 1.0 (absolutely certain).
- `is_blocked`: The probability the website has blocked the script.
    - range: 0.0 (not possible) to 1.0 (absolutely certain).

- `success`: The probability the desired task has been completed successfully.
    - range: 0.0 (not possible) to 1.0 (absolutely certain).
- `future_success`: The probability the script would complete its task if given more time.
    - range: 0.0 (not possible) to 1.0 (absolutely certain).

- `reasoning_is_correct`: The probability that all steps of reasoning produced by the script are correct.
    - range: 0.0 (not possible) to 1.0 (absolutely certain).

## Formatting The Response

Format your task in the following JSON schema:

```json
{
    "proposed_task": str | null,
    "task_is_feasible": float | null,
    "estimated_difficulty": float | null,
    "estimated_steps": int | null
}
```

Here is what each key means:

- `proposed_task`: Next task to assign the script, subject to the following guidelines:
    1. Provide a realistic, and specific task that a hypothetical user might want to accomplish on this website.
    2. Must not require making an account, logging in, or submitting personal information.
    3. Must not require external knowledge, beyond what a typical user might know.
    4. Must not involve illegal, harmful, or unethical activities.
    5. Must not involve any inappropriate, offensive, or mature content.
    6. If you are unable to satisfy all the guidelines, set all keys to `null`.

- `task_is_feasible`: The probability the proposed task is feasible on this website using the script.
    - range: 0.0 (not possible) to 1.0 (absolutely certain).
- `estimated_difficulty`: The estimated difficulty of the proposed task for the script.
    - range: 0.0 (easy) to 1.0 (difficult).
- `estimated_steps`: The estimated number of actions to complete the task.

Here are some example tasks for inspiration:

- `awg-fittings.com`: What is the C-to-C Hose-Shut-Off Valve length in mm?
- `biodiversitylibrary.org`: Open a scanned copy of 'The Angora cat; how to breed train and keep it'.
- `scholar.google.com`: How many citations does the Generative adversarial nets paper have?
- `wiktionary.org`: What is the definition and etymology of the word 'serendipity'?

Thanks for helping me direct the script, please follow the instructions carefully. Start your response with an analysis for how previous runs can be improved, followed by a step-by-step breakdown of your proposed task and associated scores, and finally, provide your task in the JSON format. Limit your response to 500 words."""


USER_PROMPT_TEMPLATE = """## Summary Of Previous Runs 

Here are previous runs of the script, including tasks, webpages, actions, and performance reviews, formatted in markdown:

{annotations}

## Formatting The Response

Enter a task in the following JSON schema:

```json
{{
    "proposed_task": str | null,
    "task_is_feasible": float | null,
    "estimated_difficulty": float | null,
    "estimated_steps": int | null
}}
```

Here is what each key means:

- `proposed_task`: Next task to assign the script, subject to the following guidelines:
    1. Provide a realistic, and specific task that a hypothetical user might want to accomplish at {target_url}.
    2. Must not require making an account, logging in, or submitting personal information.
    3. Must not require external knowledge, beyond what a typical user might know.
    4. Must not involve illegal, harmful, or unethical activities.
    5. Must not involve any inappropriate, offensive, or mature content.
    6. If you are unable to satisfy all the guidelines, set all keys to `null`.

- `task_is_feasible`: The probability the proposed task is feasible at {target_url} using the script.
    - range: 0.0 (not possible) to 1.0 (absolutely certain).
- `estimated_difficulty`: The estimated difficulty of the proposed task for the script.
    - range: 0.0 (easy) to 1.0 (difficult).
- `estimated_steps`: The estimated number of actions to complete the task.

Here are some example tasks for inspiration:

- `awg-fittings.com`: What is the C-to-C Hose-Shut-Off Valve length in mm?
- `biodiversitylibrary.org`: Open a scanned copy of 'The Angora cat; how to breed train and keep it'.
- `scholar.google.com`: How many citations does the Generative adversarial nets paper have?
- `wiktionary.org`: What is the definition and etymology of the word 'serendipity'?

Start your response with an analysis for how previous runs can be improved, followed by a step-by-step breakdown of your proposed task and associated scores, and finally, provide your task in the JSON format. Limit your response to 500 words."""


class JsonTaskParser(BaseTaskParser):
    """Implements a parser for converting text generated by an LLM into a
    task for an LLM agent to attempt to complete using a web browser,
    returns a BrowserJudgment instance parsed from the response.

    Attributes:

    system_prompt: str
        Depending on the task representation, this system prompt
        instructs the LLM on how to generate tasks in the corresponding format,
        such as JSON-based tasks, YAML format, etc.

    user_prompt_template: str
        A template string that is used to generate a user prompt for the LLM,
        and has format keys for `annotations` which represents
        previous task runs and judgments produced by the LLM judge.

    """

    system_prompt = SYSTEM_PROMPT
    user_prompt_template = USER_PROMPT_TEMPLATE

    def parse_task(self, response: str) -> BrowserTaskProposal | BrowserStatus:
        """Parse a task proposal string produced by an LLM, and return a
        BrowserTaskProposal object that contains the proposed task,
        and additional metadata about the task feasibility, and difficulty.

        Arguments:

        response: str
            The response from an LLM that contains a task proposal in a code block,
            which will be parsed into a BrowserTaskProposal object.
        
        Returns:

        BrowserTaskProposal | PlaywrightStatus
            The parsed task proposal, or a BrowserStatus object that
            represents a failed parsing attempt.
        
        """
        
        match = TASK_PATTERN.search(response)

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
            "proposed_task" in response_dict and
            "task_is_feasible" in response_dict and
            "estimated_difficulty" in response_dict and
            "estimated_steps" in response_dict
        )

        if not has_required_keys:

            return BrowserStatus.ERROR
        
        proposed_task = response_dict["proposed_task"]
        task_is_feasible = response_dict["task_is_feasible"]
        estimated_difficulty = response_dict["estimated_difficulty"]
        estimated_steps = response_dict["estimated_steps"]
        
        keys_right_type = (
            (isinstance(proposed_task, str) and (len(proposed_task) > 0)) and
            (isinstance(task_is_feasible, float) or isinstance(task_is_feasible, int)) and
            (isinstance(estimated_difficulty, float) or isinstance(estimated_difficulty, int)) and
            (isinstance(estimated_steps, float) or isinstance(estimated_steps, int))
        )

        if not keys_right_type:

            return BrowserStatus.ERROR
        
        task_dict = {
            "proposed_task": str(proposed_task),
            "task_is_feasible": float(task_is_feasible),
            "estimated_difficulty": float(estimated_difficulty),
            "estimated_steps": int(estimated_steps)
        }
        
        browser_task = BrowserTaskProposal(
            **task_dict,
            response = response,
            matched_response = matched_response
        )

        return browser_task
