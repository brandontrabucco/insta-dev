from insta.utils import (
    BrowserStatus,
    EnvError,
    ServerError,
    ERROR_TO_MESSAGE
)

from insta.configs.browser_config import (
    BrowserConfig,
    DEFAULT_BROWSER_CONFIG,
    get_browser_config
)

from insta.client import (
    BrowserClient
)

from insta.observation_processors import (
    OBSERVATION_PROCESSORS
)

from insta.action_parsers import (
    ACTION_PARSERS
)

from typing import Dict, Tuple
from PIL import Image

from dataclasses import asdict
from collections import namedtuple

import random


ADJECTIVES = [
    'abject', 'aboard', 'adoring', 'affected', 'alert', 'aloof', 'amazed', 
    'amused', 'annoyed', 'anxious', 'ardent', 'artistic', 'ashamed', 'awed', 
    'betrayed', 'blissful', 'boastful', 'bored', 'brainy', 'bubbly', 
    'cautious', 'cheerful', 'chic', 'cocky', 'content', 'cruel', 'crummy', 
    'crushed', 'cultured', 'curious', 'cynical', 'dear', 'debonair', 
    'decimal', 'dejected', 'dopey', 'dreadful', 'dreary', 'eager', 'ecstatic', 
    'empathic', 'empty', 'enraged', 'envious', 'euphoric', 'exacting', 
    'excited', 'excluded', 'fervent', 'finicky', 'fond', 'forsaken', 'giddy', 
    'gleeful', 'gloomy', 'goofy', 'grizzled', 'grudging', 'grumpy', 'guilty', 
    'guttural', 'holistic', 'humorous', 'hushed', 'imported', 'innocent', 
    'insecure', 'jealous', 'joyful', 'jubilant', 'jumpy', 'kind', 'lazy', 
    'lovesick', 'lying', 'mad', 'mellow', 'merciful', 'mere', 'mild', 'morbid', 
    'murky', 'needful', 'needy', 'obsessed', 'offended', 'outlying', 
    'pacified', 'panicky', 'peaceful', 'pesky', 'pitiful', 'pleased', 'plucky', 
    'prideful', 'puzzled', 'relieved', 'resolved', 'sad', 'scornful', 'selfish', 
    'shameful', 'sheepish', 'shy', 'similar', 'sincere', 'solemn', 'solid', 
    'somber', 'sore', 'spirited', 'stressed', 'sugary', 'superior', 'taut', 
    'thrifty', 'thrilled', 'troubled', 'trusting', 'truthful', 'unhappy', 
    'vengeful', 'wakeful', 'weary', 'winged', 'worldly', 'wornout', 'worried', 
    'wrathful', 'yearning', 'zesty', 'zippy', 'zoned', 'zooey'
]


NAMES = [
    'abalone', 'antelope', 'apples', 'apricots', 'baboon', 'bagels', 'basmati', 
    'bass', 'bittern', 'boa', 'boars', 'bobolink', 'buck', 'burritos', 
    'bustard', 'buzzard', 'cake', 'camel', 'cardinal', 'caribou', 'caviar', 
    'chamois', 'cheese', 'cheetah', 'chile', 'chough', 'chowder', 'clam', 
    'coati', 'cockatoo', 'coconut', 'cod', 'cordial', 'cow', 'crackers', 
    'crane', 'cur', 'curlew', 'dingo', 'dinosaur', 'dotterel', 'doughnut', 
    'dove', 'doves', 'dunbird', 'eagle', 'eggs', 'eland', 'falcon', 'ferret', 
    'fish', 'flamingo', 'garlic', 'gatorade', 'gelding', 'gnu', 'granola', 
    'hare', 'hawk', 'heron', 'hoopoe', 'hyena', 'icecream', 'iguana', 'jaguar', 
    'jerky', 'kitten', 'lapwing', 'lard', 'lemur', 'leopard', 'lion', 'lizard', 
    'llama', 'locust', 'lollies', 'macaw', 'mackerel', 'magpie', 'mallard', 
    'mandrill', 'mare', 'meerkat', 'moth', 'muesli', 'mussel', 'oatmeal', 
    'ocelot', 'oil', 'orange', 'oryx', 'otter', 'owl', 'paella', 'pear', 
    'pepper', 'pie', 'piglet', 'plover', 'polenta', 'ponie', 'porpoise', 
    'poultry', 'pretzels', 'pudding', 'pup', 'quiche', 'raisins', 'rat', 'relish', 
    'rhino', 'rice', 'ruffs', 'salami', 'salt', 'sardines', 'sausage', 'seafowl', 
    'seagull', 'seahorse', 'shads', 'sheep', 'smelt', 'snail', 'snipe', 'stork', 
    'swift', 'syrup', 'tacos', 'teal', 'termite', 'thrush', 'thrushe', 'tomatoe', 
    'tortoise', 'toucan', 'truffle', 'tuna', 'unicorn', 'venison', 'viper', 
    'wasp', 'weaver', 'whiting', 'widgeon', 'wigeon', 'wildfowl', 'zebra'
]


SESSION_NAMES = [
    '{}-{}'.format(a, b)
    for a in ADJECTIVES for b in NAMES
]


InstaToolOutput = namedtuple(
    "InSTAToolOutput",
    ["session_id", "processed_text", "screenshot"]
)


def create_new_session(
    base_config: BrowserConfig = DEFAULT_BROWSER_CONFIG,
    playwright_url: str = "http://localhost:{port}",
    playwright_port: int = 3000,
    playwright_workers: int = 8,
    observation_processor: str = "markdown",
    action_parser: str = "javascript",
    browser_kwargs: dict = None,
    context_kwargs: dict = None,
) -> Dict[str, any]:
    """Creates a new browsing session by connecting to the server
    and initializing a Playwright environment with the specified
    configuration settings, returning the session data.

    Returns:

    Dict[str, any]
        A dictionary containing the client, configuration, observation
        processor, action parser, and candidates.
    
    """

    config_kwargs = asdict(base_config)
    config_kwargs.update({
        "playwright_url": playwright_url,
        "playwright_port": random.randint(
            playwright_port,
            playwright_port + 
            playwright_workers - 1
        )
    })

    config = get_browser_config(
        **config_kwargs
    )   

    client = BrowserClient(
        config = config
    )

    observation_type = observation_processor
    action_type = action_parser

    observation_processor = OBSERVATION_PROCESSORS[observation_processor]()
    action_parser = ACTION_PARSERS[action_parser]()

    start_status = client.start(
        browser_kwargs = browser_kwargs,
        context_kwargs = context_kwargs
    )

    if start_status is BrowserStatus.ERROR:

        return ERROR_TO_MESSAGE[EnvError.START_ERROR]

    if isinstance(start_status, ServerError):
        
        return start_status.message
    
    return {
        "client": client,
        "config": config,
        "observation_type": observation_type,
        "action_type": action_type,
        "observation_processor": observation_processor,
        "action_parser": action_parser
    }


def return_error(
    error: EnvError | ServerError
) -> Tuple[None, str, None]:
    """Returns an informative error message ot the Gradio interface,
    and sets the session ID, and the screenshot to null.

    Arguments:

    error : EnvError | ServerError
        The error code or message to return.

    Returns:

    Tuple[None, str, None]
        A tuple containing a null session ID to reset the session ID,
        the error message, and a null screenshot.

    """

    error_message = (
        ERROR_TO_MESSAGE[error] 
        if isinstance(error, EnvError) else
        error.message
    )

    return InstaToolOutput(
        session_id = None,
        processed_text = error_message,
        screenshot = None
    )


def interact_with_browser(
    session_id: str = None, 
    url: str = None,
    action: str = None,
    active_sessions: Dict[str, any] = {},
    shorter_session_id: bool = True,
    base_config: BrowserConfig = DEFAULT_BROWSER_CONFIG,
    playwright_url: str = "http://localhost:{port}",
    playwright_port: int = 3000,
    playwright_workers: int = 8,
    observation_processor: str = "markdown",
    action_parser: str = "javascript",
    browser_kwargs: dict = None,
    context_kwargs: dict = None,
) -> Tuple[str, str, Image.Image]:
    """Entry point for the Gradio application, accepts the current browsing 
    session ID, the URL to open, and the next action to perform via 
    a snippet of Javascript code to modules in the Playwright Node.js API.

    Arguments:

    session_id : str
        The current browsing session ID, if any.

    url : str
        The URL to open in the browser.

    action : str
        The action to perform via a snippet of Javascript code, or JSON
        targetting a module in the Playwright Node.js API.

    active_sessions : Dict[str, any]
        A dictionary containing the active browsing sessions.
    
    Returns:

    Tuple[str, str, Image.Image]
        A tuple containing the assigned session ID, the processed webpage,
        and the screenshot of the webpage.
    
    """
    
    # ensure that all inputs are strings and strip any whitespace
    session_id = (session_id or "").strip()
    url = (url or "").strip()
    action = (action or "").strip()

    # handle case when an action is provided but there is no session
    if len(session_id) == 0 and len(action) > 0:

        return return_error(ServerError(
            message = "Please provide a valid session ID for actions. "
            "This can be obtained by first loading a URL.",
            status_code = 400  # placeholder for a bad request
        ))
    
    # handle case the requested session does not exist
    elif len(session_id) > 0 and session_id not in active_sessions:

        return return_error(EnvError.START_ERROR)
    
    # handle case when the sessionis correct and exists
    elif session_id in active_sessions:

        session_data = active_sessions[session_id]

        config = session_data["config"]
        client = session_data["client"]

        observation_type = session_data["observation_type"]
        action_type = session_data["action_type"]

        observation_processor = session_data["observation_processor"]
        action_parser = session_data["action_parser"]

    # handle case when a new session is requested
    elif len(session_id) == 0:

        session_data = create_new_session(
            base_config = base_config,
            playwright_url = playwright_url,
            playwright_port = playwright_port,
            playwright_workers = playwright_workers,
            observation_processor = observation_processor,
            action_parser = action_parser,
            browser_kwargs = browser_kwargs,
            context_kwargs = context_kwargs
        )

        if isinstance(session_data, str):

            return return_error(ServerError(
                message = session_data,
                status_code = 400
            ))
        
        elif isinstance(session_data, dict):

            client = session_data["client"]
            config = session_data["config"]

            observation_type = session_data["observation_type"]
            action_type = session_data["action_type"]

            observation_processor = session_data["observation_processor"]
            action_parser = session_data["action_parser"]

        session_id = client.session_id
        if shorter_session_id:
            session_id = random.choice(SESSION_NAMES)

        active_sessions[session_id] = session_data
        
    if len(action) > 0:  # take an action

        action = action_parser.parse_action(
            """Here is the action:\n\n```{type}\n{action}\n```"""
            .format(type = action_type, action = action)
        )

        if action is BrowserStatus.ERROR:

            return return_error(EnvError.ACTION_PARSE_ERROR)

        action_status = client.action(
            function_calls = action.function_calls
        )

        if action_status is BrowserStatus.ERROR:

            return return_error(EnvError.ACTION_FAILED_ERROR)

        if isinstance(action_status, ServerError):

            return return_error(action_status)

    elif len(url) > 0 and url.startswith("http"):  # load a URL

        goto_status = client.goto(url = url)

        if goto_status is BrowserStatus.ERROR:

            return return_error(EnvError.GOTO_ERROR)

        if isinstance(goto_status, ServerError):

            return return_error(goto_status)
        
    obs = client.observation()

    if obs is BrowserStatus.ERROR:

        return return_error(EnvError.PROCESSING_ERROR)
    
    if isinstance(obs, ServerError):
        
        return return_error(obs)
    
    obs = observation_processor.process(
        obs, restrict_viewport = config.restrict_viewport,
        require_visible = config.require_visible,
        require_frontmost = config.require_frontmost,
        remove_pii = config.remove_pii
    )

    return InstaToolOutput(
        session_id = session_id,
        processed_text = obs.processed_text,
        screenshot = obs.screenshot
    )
