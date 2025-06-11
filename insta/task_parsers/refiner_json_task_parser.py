from insta.task_parsers.json_task_parser import (
    JsonTaskParser
)


SYSTEM_PROMPT = """You are helping me refine tasks for a language model agent that interacts with and navigates live webpages. I instructed the agent to complete an initial task, and I will share a sequence of webpages produced by the agent as it explores the website.

## Your Instructions

Based on the agent's trajectory, help me refine the task to be precise, straightforward to evaluate, and closer in phrasing to a real user.

You will provide a task as JSON in a fenced code block:

```json
{
    "proposed_task": str,
    "steps": List[str],
    "criteria": List[str]
}
```

Tasks have the following components:

- `proposed_task`: A refined 50 word task that a real user may want to accomplish on the website.
- `steps`: Refined steps in an efficient trajectory that completes the task.
- `criteria`: Refined answers and criteria to determine if the agent completes the task.

Tasks must adhere to the following guidelines:

- Must not require logging in, or making an account.
- Must not require making a purchase, booking, or placing an order.
- Must not require creating, deleting, or modifying any posts, articles, or webpages.

## Example Tasks

I've prepared some examples to inspire your task design.

### roofingcalc.com

In this example, we explored `roofingcalc.com` and saw a roofing calculator tool that accepts several parameters, including Roof length, Roof width, Roof slope, Roof difficulty, Tear-off/disposal, Number of levels, Skylights, Chimneys, Ridge-vent, and Roofing materials.

```json
{
    "proposed_task": "Using the roofing calculator, estimate the cost of a new roof for a 50ft length and 30ft width, with a steep 12/12 slope and 'Difficult and cutup' complexity, using 'Copper Panels' as the material. Assume no tear-off, a single-story house, 2 skylights, 0 chimneys, and 20ft of ridge-vent. State the estimated cost.",
    "steps": [
        "Navigate to 'roofingcalc.com'",
        "Locate the 'Roofing Estimate Calculator'",
        "Input '50' for Roof length",
        "Input '30' for Roof width",
        "Select '12/12 (steep)' for Roof slope",
        "Select 'Difficult and cutup' for Roof difficulty",
        "Select 'No tear-off' for Tear-off/disposal",
        "Select 'Single-story' for Number of levels",
        "Input '2' for Skylights",
        "Input '0' for Chimneys",
        "Input '20' for Ridge-vent",
        "Select 'Copper Panels' for Roofing materials",
        "Click the 'Calculate' button",
        "Identify and extract the estimated cost displayed after the calculation"
    ],
    "criteria": [
        "Successfully input all specified parameters into the calculator.",
        "Select 'Copper Panels' as the roofing material.",
        "Successfully trigger the calculation and identify the resulting estimated cost.",
        "State the estimated cost for the specified parameters."
    ]
}
```

### mortgage.com

In this example, we explored `mortgage.com` and saw a mortgage calculator that can be used to determine what annual interest rate would result in a specific total monthly payment.

```json
{
    "proposed_task": "Adjust the mortgage calculator to determine what annual interest rate would result in a total monthly payment of approximately $2,800 for a $400,000 home with a 20% down payment, assuming annual property taxes of $600, annual homeowners insurance of $2,000, and monthly HOA fees of $146. Provide the interest rate and the exact calculated monthly payment.",
    "steps": [
        "Navigate to mortgage.com's monthly payment calculator.",
        "Set the Home Price to $400,000.",
        "Set the Down Payment to 20%.",
        "Set the Annual Property Taxes to $600.",
        "Set the Annual Homeowners Insurance to $2,000.",
        "Set the Monthly HOA Fees to $146.",
        "Iteratively adjust the Interest Rate to achieve a total estimated monthly payment of approximately $2,800.",
        "Once the target payment is reached, extract the final Interest Rate and the exact calculated monthly payment."
    ],
    "criteria": [
        "The Home Price is set to $400,000.",
        "The Down Payment is set to 20%.",
        "The Annual Property Taxes are set to $600.",
        "The Annual Homeowners Insurance is set to $2,000.",
        "The Monthly HOA Fees are set to $146.",
        "The final estimated monthly payment is within a reasonable range (+/- $10) of $2,800.",
        "State the final interest rate used to achieve the target payment.",
        "State the exact calculated monthly payment at that interest rate."
    ]
}
```

### giving.umw.edu

In this example, we explored `giving.umw.edu` and saw a form for submitting a one-time gift to the University of Mary Washington, which requires certain personal information.

```json
{
    "proposed_task": "Make a one-time gift of $25 to the Fund for Mary Washington, providing the following personal information: Title: Mr., First Name: John, Last Name: Doe, Street Address 1: 123 Main St, City: Fredericksburg, State: Virginia, Zip: 22401, Country: United States, Phone Area Code: 540, Phone Number: 5551234. Do not actually submit the payment.",
    "steps": [
        "Navigate to giving.umw.edu",
        "Set the donation amount to $25 in the 'Amount' field (id: 145)",
        "Confirm 'Fund for Mary Washington' is selected as the designation",
        "Select 'Mr.' for the 'Title' field (id: 955)",
        "Fill 'John' in the 'First Name' field (id: 972)",
        "Fill 'Doe' in the 'Last Name' field (id: 989)",
        "Fill '123 Main St' in the 'Street Address 1' field (id: 1095)",
        "Fill 'Fredericksburg' in the 'City' field (id: 1112)",
        "Select 'Virginia' from the 'State' dropdown (id: 1120)",
        "Fill '22401' in the 'Zip' field (id: 1208)",
        "Select 'United States' from the 'Country' dropdown (id: 1217)",
        "Fill '540' in the 'Phone Area Code' field (id: 1486)",
        "Fill '5551234' in the 'Phone Number' field",
        "State that all required fields for the donation form have been filled, but do not click 'Finish My Gift' or proceed to payment."
    ],
    "criteria": [
        "The donation amount is set to $25.",
        "The 'Fund for Mary Washington' is selected.",
        "All 'Your Information' required fields (Title, First Name, Last Name, Street Address 1, City, State, Zip, Country, Phone Area Code, Phone Number) are filled with the specified information.",
        "Do not click 'Finish My Gift' or proceed to any payment processing page."
    ]
}
```

### odetterestaurant.com

In this example, we explored `odetterestaurant.com` and saw a 'Reservations' page, which lists policies for dietary accommodations, birthdays, a deposit requirement, cancellations, and rescheduling.

```json
{
    "proposed_task": "I want to make a dinner reservation for 4 people at Odette, and one of my guests has a severe dairy allergy. I also want to request a birthday cake for the table. What are the key policies I need to be aware of regarding my guest's allergy, the cake request, and any deposit or cancellation rules for this reservation?",
    "steps": [
        "Navigate to 'odetterestaurant.com'",
        "Go to the 'Reservations' page",
        "Identify the policy regarding dairy allergies and other dietary accommodations",
        "Find the policy for requesting a birthday cake, including notice period and cost",
        "Locate the deposit requirement per person for dinner reservations",
        "Determine the cancellation or rescheduling policy and associated timeframe",
        "Synthesize all relevant policies into a concise answer"
    ],
    "criteria": [
        "State that Odette is unable to accommodate guests with dairy allergies or intolerance.",
        "State that cakes require a 72-hour notice and cost SGD78++.",
        "Confirm a deposit of SGD200 per person is required for dinner reservations.",
        "State that all reservations are final and non-refundable, but changes can be made at least 72 hours prior to the reservation date."
    ]
}
```

## Formatting Your Response

Write a 300 word analysis that establishes how the task can be refined, and synthesizes relevant information we saw on the website. After your response, provide a task as JSON in a fenced code block."""


USER_PROMPT_TEMPLATE = """## Refine The Task For This Website

You are viewing the agent's trajectory:

{summary}

## Your Instructions

Based on the agent's trajectory, help me refine the task to be precise, straightforward to evaluate, and closer in phrasing to a real user.

You will provide tasks as JSON in a fenced code block:

```json
{{
    "proposed_task": str,
    "steps": List[str],
    "criteria": List[str]
}}
```

Tasks have the following components:

- `proposed_task`: A refined 50 word task that a real user may want to accomplish on {website}.
- `steps`: Refined steps in an efficient trajectory that completes the task.
- `criteria`: Refined answers and criteria to determine if the agent completes the task.

## Formatting Your Response

Write a 300 word analysis that establishes how the task can be refined, and synthesizes relevant information we saw on {website}. After your response, provide a task as JSON in a fenced code block."""


class RefinerJsonTaskParser(JsonTaskParser):
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
