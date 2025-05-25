from insta.utils import (
    BrowserStatus
)

from insta.configs.task_proposer_config import (
    BrowserTaskProposal
)

from insta.task_parsers.task_parser import (
    BaseTaskParser
)

import re
import json


TASK_PATTERN = re.compile(
    r"```json\n(?P<json>.*)\n```",
    re.DOTALL
)


SYSTEM_PROMPT = """You are helping me design tasks for a language model agent that interacts with and navigates live webpages. I instructed the agent to complete an initial task on a website, and I will share a sequence of webpages, actions, and a final success score produced by the agent.

## Your Instructions

Based on the agent's trajectory, you are helping me design challenging tasks as if you were a real user on the website.

You will provide tasks as JSON in a fenced code block:

```json
{
    "proposed_task": str,
    "steps": List[str],
    "criteria": List[str]
}
```

Tasks have the following components:

- `proposed_task`: A challenging task that a real user wants to accomplish on the website.
- `steps`: Steps in the most efficient trajectory that completes the task.
- `criteria`: Rigorous success criteria to determine if the agent completes the task.

## Example Tasks

I've prepared some examples to inspire your task design.

### awg-fittings.com

In this example, we explored the website 'awg-fittings.com' and saw a product listing for the 'C-to-C Hose-Shut-Off Valve', which was listed as having a length of '237 mm' in its product specifications.

```json
{
    "proposed_task": "What is the C-to-C Hose-Shut-Off Valve length in mm on AWG Fittings?",
    "steps": [
        "Navigate to 'awg-fittings.com'",
        "Open the product catalog for fittings",
        "Locate the product listing for the C-to-C Hose-Shut-Off Valve",
        "Find the product length in mm, and respond with that length in the answer"
    ],
    "criteria": [
        "Cites the specific length of '237 mm' for this product"
    ]
}
```

### biodiversitylibrary.org

In this example, we explored 'biodiversitylibrary.org', saw a document titled 'The Angora cat; how to breed train and keep it', and the website had an embedded PDF reader agents can use to open the document.

```json
{
    "proposed_task": "Open a scanned copy of 'The Angora cat; how to breed train and keep it'.",
    "steps": [
        "Navigate to 'biodiversitylibrary.org'",
        "Search for 'The Angora cat; how to breed train and keep it' in the search bar",
        "Click on the title of the document in the search results",
        "Confirm the correct document is displayed in an embedded PDF reader"
    ],
    "criteria": [
        "Displays the correct document in an embedded PDF reader"
    ]
}
```

### scholar.google.com

In this example, we explored 'scholar.google.com' and saw a paper titled 'Generative Adversarial Networks' with a citation count of '80613' in the search results (as of today's date).

```json
{
    "proposed_task": "How many citations does the paper 'Generative Adversarial Networks' have?",
    "steps": [
        "Navigate to 'scholar.google.com'",
        "Search for 'Generative Adversarial Networks' in the search bar",
        "Locate the correct paper in the search results",
        "Find an up-to-date citation count, and respond with that count in the answer"
    ],
    "criteria": [
        "Has an up-to-date citation count, which is '80613' as of April 2025",
        "The answer matches the citation count for the correct paper in the search results"
    ]
}
```

### wiktionary.org

In this example, we explored 'wiktionary.org' and saw a webpage discussing the definition and etymology of the word 'serendipity', which is derived from 'Serendip' or 'Serendib', and was coined by English writer and politician Horace Walpole in 1754.

```json
{
    "proposed_task": "What is the definition and etymology of the word 'serendipity'?",
    "steps": [
        "Navigate to 'wiktionary.org'",
        "Search for 'serendipity' in the search bar",
        "Find the definition and etymology sections of the 'serendipity' page",
        "Summarize the contents of these sections in the answer"
    ],
    "criteria": [
        "Mentions the word is derived from Serendip (or Serendib)",
        "States it was coined by English writer and politician Horace Walpole in 1754"
    ]
}
```

## Formatting Your Response

Write a 300 word analysis that establishes what real users want to accomplish on the website, and highlights key information we saw on the website. After your response, provide your task as JSON in a fenced code block."""


USER_PROMPT_TEMPLATE = """## Design A Task For This Website

You are viewing the agent's trajectory.

{summary}

## Your Instructions

Based on the agent's trajectory, you are helping me design challenging tasks as if you were a real user on {website}.

You will provide tasks as JSON in a fenced code block:

```json
{{
    "proposed_task": str,
    "steps": List[str],
    "criteria": List[str]
}}
```

Tasks have the following components:

- `proposed_task`: A challenging task that a real user wants to accomplish on {website}.
- `steps`: Steps in the most efficient trajectory that completes the task.
- `criteria`: Rigorous success criteria to determine if the agent completes the task.

## Formatting Your Response

Write a 300 word analysis that establishes what real users want to accomplish on {website}, and highlights key information we saw on {website}. After your response, provide your task as JSON in a fenced code block."""


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
    
            raise ValueError(
                "Failed to parse task"
            )

        matched_response = match.group("json")
        response_dict = json.loads(
            matched_response
        )
        
        has_required_keys = (
            "proposed_task" in response_dict and
            "steps" in response_dict and
            "criteria" in response_dict
        )

        if not has_required_keys:
    
            raise ValueError(
                "Failed to parse task"
            )
        
        proposed_task = response_dict["proposed_task"]
        steps = response_dict["steps"]
        criteria = response_dict["criteria"]
        
        keys_right_type = (
            (isinstance(proposed_task, str) and (len(proposed_task) > 0)) and
            (isinstance(steps, list) and (len(steps) > 0)) and
            (isinstance(criteria, list) and (len(criteria) > 0))
        )

        if not keys_right_type:
    
            raise ValueError(
                "Failed to parse task"
            )
        
        task_dict = {
            "proposed_task": str(proposed_task),
            "steps": list(steps),
            "criteria": list(criteria),
        }
        
        browser_task = BrowserTaskProposal(
            **task_dict,
            response = response,
            matched_response = matched_response
        )

        return browser_task
