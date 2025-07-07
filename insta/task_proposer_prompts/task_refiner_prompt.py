from insta.task_proposer_prompts.base_task_proposer_prompt import (
    BaseTaskProposerPrompt
)


SYSTEM_PROMPT = """You are helping me instruct a language model agent that interacts with and navigates live webpages. We instructed the agent to complete an initial task, and I will share a sequence of webpages visited by the agent during its operation.

## Your Instructions

Help me refine the task, steps and criteria to raise the difficulty, while balancing the agent's capacity to successfully complete the task.

You will provide a task as JSON in a fenced code block:

```json
{
    "proposed_task": str,
    "steps": List[str],
    "criteria": List[str]
}
```

Tasks have the following components:

- `proposed_task`: Instruct the agent to complete a task for you as if you are a real user that wants help on the website.
- `steps`: Precise steps in an efficient trajectory that completes the task.
- `criteria`: Ground truth answers and criteria to determine if the agent completes the task.

Tasks must adhere to the following guidelines:

- Must not require logging in, or making an account.
- Must not require making a purchase, booking, or placing an order.
- Must not require creating, deleting, or modifying any posts, articles, or webpages.

## Example Tasks

I've prepared some examples to inspire your task design.

### `liveevents.iadb.org`

In this example, we explored `liveevents.iadb.org` and saw an event page for the IDB Annual Meetings, which includes a list of the official hotels and instructions for official delegations.

```json
{
    "proposed_task": "I'm attending the IDB Annual Meetings and need to find accommodation. Please provide the address and phone number for the 'Pullman Santiago Vitacura' and 'Double Tree by Hilton' hotels. Additionally, what specific details do official delegations need to provide to access their special hotel block?",
    "steps": [
        "Navigate to 'https://liveevents.iadb.org'.",
        "Click on the 'Hotels' link in the navigation menu.",
        "Locate 'Pullman Santiago Vitacura' in the 'OFFICIAL HOTELS FOR THE ANNUAL MEETINGS' list and extract its address and telephone number.",
        "Locate 'Double Tree by Hilton' in the same list and extract its address and telephone number.",
        "Read the instructions under 'HOTELS FOR OFFICIAL DELEGATIONS' to identify the required information for accessing the special hotel block.",
        "State the addresses and telephone numbers for both hotels and the required information for official delegations."
    ],
    "criteria": [
        "The agent successfully navigates to the 'Hotels' page on 'liveevents.iadb.org'.",
        "The address for Pullman Santiago Vitacura is stated as 'Avenida Vitacura 3201 Vitacura, 7630578 Santiago, Chile'.",
        "The telephone number for Pullman Santiago Vitacura is stated as '+56 2 2944 7800'.",
        "The address for Double Tree by Hilton is stated as 'Avenida Vitacura 2727, Las Condes Santiago, Chile'.",
        "The telephone number for Double Tree by Hilton is stated as '+56 2 2587 7000'.",
        "The agent states that official delegations need to include 'the name of your country' and 'the code (included in the invitation letters to the Governors)' to access the special hotel block."
    ]
}
```

### `boldtcastle.com`

In this example, we explored `boldtcastle.com` and saw a page with information about visiting Boldt Castle, including operating dates, admission prices, and how to get to Heart Island.

```json
{
    "proposed_task": "Help me plan a visit to Boldt Castle in 2025 with one adult and one 6-year-old. Please provide the operating dates and hours for the 2025 season, the admission cost for just the castle for both of us, how to get to Heart Island, and the best phone number for general inquiries.",
    "steps": [
        "Navigate to 'boldtcastle.com'.",
        "Click on the 'Visiting' link.",
        "Click on the 'Plan Your Visit' link.",
        "Identify the 2025 season operating dates and hours for Boldt Castle.",
        "Find the Boldt Castle-only admission price for an adult (13+ years).",
        "Find the Boldt Castle-only admission price for a child aged 6 (5-12 years).",
        "Locate information on how to get to Boldt Castle on Heart Island.",
        "Find the general inquiry phone number for Boldt Castle.",
        "Synthesize all collected information into a concise answer."
    ],
    "criteria": [
        "State the 2025 season operating dates and hours for Boldt Castle as May 10 - October 13, 10:30 AM - 6:30 PM.",
        "State the adult admission price for Boldt Castle only as $13.50.",
        "State the admission price for a child aged 6 for Boldt Castle only as $9.50.",
        "Provide the physical location of Boldt Castle (Heart Island, Alexandria Bay, New York) and mention it's only accessible by water.",
        "Provide the general inquiry phone number for Boldt Castle as 315-482-9724."
    ]
}
```

### `visitwestchesterny.com`

In this example, we explored `visitwestchesterny.com` and saw a page that lists various coffee houses in Westchester County, including their names, addresses, and links to their descriptions.

```json
{
    "proposed_task": "Find a cozy coffee shop in Westchester County. Navigate to the 'Coffee Houses' section on the Visit Westchester NY website. Find a coffee shop described as 'cozy' and provide its name, full address, and the exact sentence from its description that indicates it is cozy.",
    "steps": [
        "Navigate to 'visitwestchesterny.com'",
        "Click on 'Things to Do'",
        "Click on 'Food and Drink'",
        "Click on 'Coffee Houses'",
        "Scroll down to view the coffee shop listings.",
        "Identify 'Altamira Cafe Bar' (or any other coffee shop) described as 'cozy'.",
        "Extract the name and address of the identified coffee shop.",
        "Click the 'Details' link for the identified coffee shop.",
        "Identify and extract the exact sentence from the description on its dedicated page that indicates it is cozy."
    ],
    "criteria": [
        "Successfully navigate to the 'Coffee Houses' page.",
        "Identify a coffee shop described as 'cozy' (e.g., 'Altamira Cafe Bar').",
        "State the name of the identified coffee shop (e.g., 'Altamira Cafe Bar').",
        "State the full address of the identified coffee shop (e.g., '245 Main St., New Rochelle, NY 10801').",
        "Successfully navigate to the 'Details' page for the identified coffee shop.",
        "Correctly state the exact sentence from the description that indicates it is cozy (e.g., 'Relax in the cozy shop or take a treat to go with piping hot espresso, a cold coffee, delicious desserts and delightful sandwiches.')."
    ]
}
```

### `odetterestaurant.com`

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

### `dottyabouticecream.co.uk`

In this example, we explored `dottyabouticecream.co.uk` and saw a form for hiring Dotty's ice cream van for corporate events, which includes fields for event details, guest count, and flavor inquiries.

```json
{
    "proposed_task": "Inquire about hiring Dotty's ice cream van for a corporate event in Manchester, M1 1AE, on August 15th, 2024, from 2 PM to 4 PM, for 100 guests. Ask if vanilla, honeycomb crunch, and mango sorbet are available. Fill out the 'Get in Touch' form with your details (Jane Doe, jane.doe@example.com, 07123456789) and note you found them via a web search. Do not submit the form.",
    "steps": [
        "Navigate to the 'Get in Touch' page on dottyabouticecream.co.uk.",
        "Fill 'Jane Doe' into the 'Name' field.",
        "Fill 'jane.doe@example.com' into the 'Email' field.",
        "Fill '07123456789' into the 'Telephone Number' field.",
        "Fill 'August 15th, 2024' into the 'Event Date' field.",
        "Fill '2 PM - 4 PM' into the 'Time of Ice Cream Service' field.",
        "Fill 'Manchester, M1 1AE' into the 'Venue Address (incl. Postcode)' field.",
        "Fill '100' into the 'Number of Expected Guests' field.",
        "Fill 'Web Search' into the 'Where Did You Hear About Dotty?' field.",
        "Fill the 'Message' field with an inquiry about the availability of 'Vanilla, Honeycomb Crunch, and Mango Sorbet' flavors for a corporate event.",
        "Confirm all specified fields are accurately filled, but do not click the 'Gimmie Ice Cream' submit button."
    ],
    "criteria": [
        "The agent successfully navigates to the 'Get in Touch' page.",
        "The 'Name' field is filled with 'Jane Doe'.",
        "The 'Email' field is filled with 'jane.doe@example.com'.",
        "The 'Telephone Number' field is filled with '07123456789'.",
        "The 'Event Date' field is filled with 'August 15th, 2024'.",
        "The 'Time of Ice Cream Service' field is filled with '2 PM - 4 PM'.",
        "The 'Venue Address (incl. Postcode)' field is filled with 'Manchester, M1 1AE'.",
        "The 'Number of Expected Guests' field is filled with '100'.",
        "The 'Where Did You Hear About Dotty?' field is filled with 'Web Search'.",
        "The 'Message' field clearly inquires about the availability of 'Vanilla, Honeycomb Crunch, and Mango Sorbet' flavors for a corporate event.",
        "The agent does not submit the form by clicking the 'Gimmie Ice Cream' button."
    ]
}
```

### `engineered.polestar.com`

In this example, we explored `engineered.polestar.com` and saw a page with information about Polestar Engineered Optimization for various Volvo models, including the 2023 Volvo XC60 with a B5 Drive-E engine.

```json
{
    "proposed_task": "Is a Polestar Engineered Optimization available for a 2023 Volvo XC60 with a B5 Drive-E engine? If so, what are the primary performance benefits, how long does the installation take, and how would I find a dealer for this service?",
    "steps": [
        "Navigate to engineered.polestar.com.",
        "Under 'Can My Volvo Be Optimised?', select 'XC' then 'New XC60' for the model.",
        "Locate and select 'XC60 B5 Drive-E AWD Automatic 2023' or 'XC60 B5 Drive-E FWD Automatic 2023' to view its optimization details.",
        "Confirm if the vehicle is 'Approved for Polestar Engineered Optimization'.",
        "Identify the primary performance benefits listed for the optimization.",
        "Determine the approximate installation time.",
        "Click on any 'Find a retailer' or 'Contact a dealer' links to see where they lead.",
        "Based on the website's information, describe how a user would find a dealer for installation.",
        "Synthesize all gathered information to answer the task."
    ],
    "criteria": [
        "Confirm that a 2023 Volvo XC60 with a B5 Drive-E engine is 'Approved for Polestar Engineered Optimization'.",
        "State the primary performance benefit as 'Power Mid-Range up to (hp) +3%' (from the specific product page) or 'Up to +15% increased mid-range power' (from the general 'Get optimisation' page).",
        "State that the installation takes 'less than 60 minutes'.",
        "Clearly state that clicking the 'Find a retailer' or 'Contact a dealer' links does not lead to a functional dealer search tool, and that users are advised to contact their local Volvo retailer directly for further questions."
    ]
}
```

### `ajga.org`

In this example, we explored `ajga.org` and saw a page with information about Performance Based Entry (PBE) Stars for junior golfers, including how they carry over to the next season and tips for maximizing tournament opportunities.

```json
{
    "proposed_task": "As a junior golfer, help me understand how my Performance Based Entry (PBE) Stars carry over to the next season on the AJGA circuit. Additionally, I'm looking for two tips from the website on how to best maximize my opportunities to play in tournaments. Can you provide this information?",
    "steps": [
        "Navigate to the 'Juniors' section of the AJGA website.",
        "Click on 'How to Play in the AJGA'.",
        "Click on the 'PBE' link within the 'How to Play in the AJGA' section.",
        "Scroll down to locate and click on the 'Important Notes and Tips' link.",
        "Identify and extract information regarding the carry-over of Performance Stars to the next season.",
        "Identify and extract two distinct tips that help players maximize their tournament playing opportunities.",
        "Synthesize the gathered information to provide a comprehensive answer."
    ],
    "criteria": [
        "Explain that Performance Stars earned in one year (e.g., 2024) are carried over to the beginning of the next season (e.g., 2025) and are combined with new membership Performance Stars based on grad year.",
        "Identify and state that 'Plan your tournament schedule early to maximize playing opportunities and prevent missed deadlines.' is a key recommendation.",
        "Identify and state that 'Qualifiers are great opportunities for all players to earn Performance Stars and build their status.' is a second key recommendation."
    ]
}
```

### `passports.gov.au`

In this example, we explored `passports.gov.au` and saw a page with a section about the documents needed to prove Australian citizenship for individuals born in Australia on or after August 20, 1986.

```json
{
    "proposed_task": "I was born in Australia on or after August 20, 1986, and am applying for my first Australian passport. What documents do I need to prove my Australian citizenship? Please list the primary document, acceptable alternatives, and any specific requirements for proving citizenship by birth based on my parents' or grandparents' status.",
    "steps": [
        "Navigate to 'passports.gov.au'.",
        "Navigate to the 'How it works' section.",
        "From 'How it works', navigate to 'Documents you need'.",
        "On the 'Documents you need' page, navigate to the 'Citizenship' section.",
        "Within the 'Citizenship' section, locate the information for individuals 'Born in Australia on or after 20 August 1986'.",
        "Identify the primary document required for proof of citizenship.",
        "Identify and list all acceptable alternative documents.",
        "Detail the specific scenarios for proving citizenship by birth, including those involving parents' or grandparents' documentation and the special case for permanent resident parents."
    ],
    "criteria": [
        "State that the primary document is the applicant's full Australian birth certificate.",
        "List an Australian citizenship certificate in the applicant's name as an acceptable alternative.",
        "List an Australian passport issued in the applicant's name on or after 1 January 2000 that was valid for at least two years as an acceptable alternative.",
        "Detail the scenario where one parent was an Australian permanent resident or citizen, specifying the required parental documents (birth certificate, passport, or citizenship certificate).",
        "Explicitly mention that if both parents were Australian permanent residents when the applicant was born, evidence of citizenship must be obtained from the Department of Home Affairs.",
        "Include the scenario involving grandparents' documents (birth certificate, passport, or citizenship certificate) if the parent was born in Australia on or after 20 August 1986."
    ]
}
```

## Formatting Your Response

Establish how the task can be refined in at most 300 words, and synthesize relevant content and features on the website in your response. After your response, provide a refined task as JSON in a fenced code block."""

USER_PROMPT_TEMPLATE = """## Initial Task & The Agent's Trajectory

Shown below is the agent's trajectory during its attempt to complete an initial task.

{summary}

## Your Instructions

Help me refine the task, steps and criteria to raise the difficulty, while balancing the agent's capacity to successfully complete the task.

You will provide a task as JSON in a fenced code block:

```json
{{
    "proposed_task": str,
    "steps": List[str],
    "criteria": List[str]
}}
```

Tasks have the following components:

- `proposed_task`: Instruct the agent to complete a task for you as if you are a real user that wants help on {website}.
- `steps`: Precise steps in an efficient trajectory that completes the task.
- `criteria`: Ground truth answers and criteria to determine if the agent completes the task.

## Formatting Your Response

Establish how the task can be refined in at most 300 words, and synthesize relevant content and features on {website} in your response. After your response, provide a refined task as JSON in a fenced code block."""


class TaskRefinerPrompt(BaseTaskProposerPrompt):
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
