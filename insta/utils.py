from dataclasses import dataclass
from typing import Dict, Any, Callable, List
from enum import Enum
from PIL import Image

import time
import traceback


BrowserStatus = Enum("PlaywrightStatus", [
    "SUCCESS",
    "NOOP",
    "ERROR",
])


EnvError = Enum("EnvError", [
    "NOT_INITIALIZED_ERROR",
    "PROCESSING_ERROR",
    "CLOSE_ERROR",
    "START_ERROR",
    "GOTO_ERROR",
    "ACTION_PARSE_ERROR",
    "ACTION_KEY_ERROR",
    "UNKNOWN_ELEMENT_ERROR",
    "ACTION_NOT_SUPPORTED_ERROR",
    "ACTION_FAILED_ERROR",
    "CANDIDATE_ERROR",
    "URL_ERROR",
])


@dataclass
class ServerError:

    status_code: int
    message: str


@dataclass
class NodeMetadata:

    backend_node_id: str = None
    candidate_key: str = None

    bounding_client_rect: dict = None
    computed_style: dict = None

    scroll_left: int = None
    scroll_top: int = None

    editable_value: str = None


NodeToMetadata = Dict[str, NodeMetadata]


@dataclass
class BrowserObservation:

    processed_text: str = None
    raw_html: str = None

    processed_image: Image.Image = None
    screenshot: Image.Image = None

    metadata: NodeToMetadata = None
    current_url: str = None


@dataclass
class FunctionCall:

    dotpath: str = None
    args: str = None


@dataclass
class BrowserAction:

    function_calls: List[FunctionCall] = None
    response: str = None
    matched_response: str = None


@dataclass
class BrowserJudgment:

    values: Dict[str, float] = None
    response: str = None
    matched_response: str = None


ERROR_TO_MESSAGE = {

    EnvError.NOT_INITIALIZED_ERROR:
    "Environment not initialized.",

    EnvError.PROCESSING_ERROR:
    "Error processing page.",

    EnvError.CLOSE_ERROR:
    "Error closing environment.",

    EnvError.START_ERROR:
    "Error starting environment.",

    EnvError.GOTO_ERROR:
    "Error navigating to URL.",

    EnvError.ACTION_PARSE_ERROR:
    "The provided action could not be parsed.",

    EnvError.ACTION_KEY_ERROR:
    "No `action_key` was provided.",

    EnvError.UNKNOWN_ELEMENT_ERROR:
    "Unable to locate the target element on the page.",

    EnvError.ACTION_NOT_SUPPORTED_ERROR: 
    "Target element does not support this action.",

    EnvError.ACTION_FAILED_ERROR:
    "Failed to perform the last action.",

    EnvError.CANDIDATE_ERROR:
    "Invalid target element ID.",

    EnvError.URL_ERROR:
    "Invalid URL.",

}


SKIP_TAGS = [
    'HTML',
    'HEAD',
    'TITLE',
    'BODY',
    'SCRIPT',
    'STYLE',
    'LINK',
    'META',
    'NOSCRIPT',
    'IFRAME',
    'FRAME',
    'FRAMESET'
]


def safe_call(
    func: Callable, *func_args: Any,
    catch_errors: bool = True,
    log_errors: bool = True,
    max_errors: int = 3,
    exponential_backoff: bool = True,
    exponential_backoff_factor: float = 1.5,
    error_class: type = Exception,
    error_callback_func: Callable = None,
    **func_kwargs: Any
) -> BrowserStatus | Any:
    """Call a function, and catch any errors that occur during the execution,
    with the option to retry the function call a specified number of times,
    with an exponential backoff delay between retries.

    Arguments:

    func: Callable
        The function to call.
    
    *func_args: Any
        The positional arguments to pass to the function.

    catch_errors: bool
        Whether to catch errors that occur during the function call.

    log_errors: bool
        Whether to log errors that occur during the function call.

    max_errors: int
        The maximum number of times to retry the function call.

    exponential_backoff: bool
        Whether to use an exponential backoff delay between retries.

    exponential_backoff_factor: float
        The factor by which to increase the delay between retries.

    error_class: type
        The error class to catch during the function call.

    error_callback_func: Callable
        A callback function to execute when an error is caught.

    **func_kwargs: Any
        The keyword arguments to pass to the function.

    Returns:

    PlaywrightStatus | Any
        The result of calling the `func` argument, or a PlaywrightStatus
        indicating whether the function call was successful,
        or an error occurred during the function call.
    
    """

    if not catch_errors: 

        return func(*func_args, **func_kwargs)

    for error_idx in range(max_errors):

        try: return func(*func_args, **func_kwargs)

        except error_class as error:

            if log_errors: print(
                traceback.format_exc()
            )

            callback_status = error_callback_func({
                "error": error, "error_idx": error_idx,
            }) if error_callback_func is not None else None

            if callback_status is BrowserStatus.ERROR:

                return BrowserStatus.ERROR
                
        if exponential_backoff: time.sleep(
            exponential_backoff_factor 
            ** error_idx
        )
            
    return BrowserStatus.ERROR
