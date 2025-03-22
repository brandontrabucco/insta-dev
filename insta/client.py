from insta.configs.browser_config import (
    BrowserConfig,
    DEFAULT_BROWSER_CONFIG,
    BrowserObservation,
    FunctionCall,
)

from insta.utils import (
    BrowserStatus,
    safe_call,
    ServerError
)

from PIL import Image
from typing import List

import requests
import base64
import io
import time


ENDPOINTS = {
    "start": "{server_url}/start?width={width}&height={height}",
    "close": "{server_url}/close?session_id={session_id}",
    "goto": "{server_url}/goto?url={url}&session_id={session_id}",
    "observation": "{server_url}/observation?session_id={session_id}",
    "action": "{server_url}/action?session_id={session_id}",
}


ClientError = (
    BrowserStatus | 
    ServerError
)


class BrowserClient(object):
    """Client for connecting to a serving running Playwright that
    manages web browsing sessions, process observations that
    contain the full state of webpages, and executing actions in
    the full Playwright API via a dotpath syntax.

    Attributes:

    config: EnvConfig
        The configuration for the Playwright environment, refer to
        insta/configs/env_config.py for more information.

    session_id: str
        Unique string identifier for the current web browsing session,
        used to identify which session on the backend server to interact with.

    initialized: bool
        Check whether the client has a valid session ID, indicating
        that a web browsing session is currently active.

    """

    def __init__(self, config: BrowserConfig = DEFAULT_BROWSER_CONFIG):
        """Client for connecting to a serving running Playwright that
        manages web browsing sessions, process observations that
        contain the full state of webpages, and executing actions in
        the full Playwright API via a dotpath syntax.

        Arguments:

        config: EnvConfig
            The configuration for the Playwright environment, refer to
            insta/configs/env_config.py for more information.
        
        """
        
        self.config = config

        self.session_id: str = None

    @property
    def initialized(self) -> bool:
        """Check whether the client has a valid session ID, indicating
        that a web browsing session is currently active.

        Returns:

        initialized: bool
            Whether the client has a valid session ID or not.
        
        """

        return self.session_id is not None

    def start(
        self, browser_kwargs: dict = None,
        context_kwargs: dict = None,
    ) -> ClientError:
        """Attempt to start a new web browsing session by connecting to
        the Playwrightserver via the /start endpoint, this will create a
        new browser and context for the session, and returns an ID
        that can be used to interact with the session.

        Arguments:

        browser_kwargs: dict
            Keyword options to use when starting the Browser instance,
            refer to the Playwright docs for more information.

        context_kwargs: dict
            Keyword options to use when starting the BrowserContext instance,
            refer to the Playwright docs for more information.

        Returns:

        PlaywrightStatus | ServerError
            The status of the start operation, or an error if the server
            failed to start a new web browsing session.
        
        """

        # close the previous session, do not stop for errors
        if self.initialized: self.close()

        endpoint = ENDPOINTS["start"].format(
            server_url = self.config.playwright_url.format(
                port = self.config.playwright_port
            ),
            width = self.config.screen_width,
            height = self.config.screen_height
        )

        json_data = None

        if browser_kwargs is not None:

            json_data = json_data or {}

            json_data.update({
                "browser_kwargs": browser_kwargs
            })

        if context_kwargs is not None:

            json_data = json_data or {}

            json_data.update({
                "context_kwargs": context_kwargs
            })

        response = safe_call(
            requests.post, endpoint, json = json_data,
            catch_errors = self.config.catch_errors,
            log_errors = self.config.log_errors,
            max_errors = self.config.max_errors
        )

        if response is BrowserStatus.ERROR:

            return BrowserStatus.ERROR

        if response.status_code != 200:

            return ServerError(
                message = response.text,
                status_code = response.status_code
            )

        self.session_id = response.text

        return BrowserStatus.SUCCESS
    
    def close(self) -> ClientError:
        """Attempt to close the current session by connecting to the
        Playwright server via the /close endpoint, this will release
        any resources that were allocated in the backend server.
        
        Returns:

        PlaywrightStatus | ServerError
            The status of the close operation, or an error if the server
            failed to close the current web browsing session.
        
        """

        if not self.initialized:

            return BrowserStatus.SUCCESS

        endpoint = ENDPOINTS["close"].format(
            server_url = self.config.playwright_url.format(
                port = self.config.playwright_port
            ),
            session_id = self.session_id
        )

        response = safe_call(
            requests.post, endpoint,
            catch_errors = self.config.catch_errors,
            log_errors = self.config.log_errors,
            max_errors = self.config.max_errors
        )

        self.session_id = None
        
        if response is BrowserStatus.ERROR:

            return BrowserStatus.ERROR

        if response.status_code != 200:

            return ServerError(
                message = response.text,
                status_code = response.status_code
            )

        return BrowserStatus.SUCCESS
    
    def goto(self, url: str) -> ClientError:
        """Attempt to navigate to a new URL by connecting to the
        Playwright server via the /goto endpoint, this will change the
        current URL of the web browsing session to the new URL.

        Arguments:

        url: str
            The URL to navigate to in the current web browsing session.

        Returns:

        PlaywrightStatus | ServerError
            The status of the goto operation, or an error if the server
            failed to navigate to the new URL.
        
        """

        time.sleep(
            self.config.delays.get("goto", 0)
            if self.config.delays is not None else 0
        )
        
        if not self.initialized:
            
            return BrowserStatus.ERROR

        endpoint = ENDPOINTS["goto"].format(
            server_url = self.config.playwright_url.format(
                port = self.config.playwright_port
            ),
            url = url,
            session_id = self.session_id
        )

        response = safe_call(
            requests.post, endpoint,
            catch_errors = self.config.catch_errors,
            log_errors = self.config.log_errors,
            max_errors = self.config.max_errors
        )

        if response is BrowserStatus.ERROR:

            return BrowserStatus.ERROR

        if response.status_code != 200:

            return ServerError(
                message = response.text,
                status_code = response.status_code
            )

        return BrowserStatus.SUCCESS
    
    def observation(self) -> (BrowserObservation | ClientError):
        """Attempt to generate an observation by connecting to the
        Playwright server via the /observation endpoint.

        Returns:

        PlaywrightObservation | PlaywrightStatus | ServerError
            The observation data, the status of the observation operation,
            or an error if the server failed to generate an observation.

        """

        time.sleep(
            self.config.delays.get("observation", 0)
            if self.config.delays is not None else 0
        )
        
        if not self.initialized:
            
            return BrowserStatus.ERROR

        endpoint = ENDPOINTS["observation"].format(
            server_url = self.config.playwright_url.format(
                port = self.config.playwright_port
            ),
            session_id = self.session_id
        )

        response = safe_call(
            requests.post, endpoint,
            catch_errors = self.config.catch_errors,
            log_errors = self.config.log_errors,
            max_errors = self.config.max_errors
        )

        if response is BrowserStatus.ERROR:

            return BrowserStatus.ERROR

        if response.status_code != 200:

            return ServerError(
                message = response.text,
                status_code = response.status_code
            )

        obs_data = response.json()

        expected_keys = [
            "raw_html",
            "screenshot",
            "metadata",
            "current_url"
        ]

        all_keys_present = all([
            key in obs_data
            for key in expected_keys
        ])

        if not all_keys_present:

            return BrowserStatus.ERROR
            
        screenshot = Image.open(io.BytesIO(
            base64.b64decode(obs_data["screenshot"])
        ))

        observation = BrowserObservation(
            raw_html = obs_data["raw_html"],
            screenshot = screenshot,
            metadata = obs_data["metadata"],
            current_url = obs_data["current_url"]
        )

        return observation
    
    def action(self, function_calls: List[FunctionCall]) -> ClientError:
        """Attempt to perform a sequence of function calls in the browsing session
        by connecting to the Playwright server via the /action endpoint.

        Arguments:

        function_calls: List[FunctionCall]
            A sequence of function calls to perform in the web browsing session,
            where each function calls is represented via a `dotpath` to a 
            function in the Playwright API, and corresponding arguments to pass.

        Returns:

        PlaywrightObservation | PlaywrightStatus | ServerError
            The observation data, the status of the observation operation,
            or an error if the server failed to generate an observation.
        
        """

        time.sleep(
            self.config.delays.get("action", 0)
            if self.config.delays is not None else 0
        )
        
        if not self.initialized:
            
            return BrowserStatus.ERROR

        endpoint = ENDPOINTS["action"].format(
            server_url = self.config.playwright_url.format(
                port = self.config.playwright_port
            ),
            session_id = self.session_id
        )

        action_json = [
            {"dotpath": x.dotpath, "args": x.args}
            for x in function_calls
        ]

        response = safe_call(
            requests.post, endpoint, json = action_json,
            catch_errors = self.config.catch_errors,
            log_errors = self.config.log_errors,
            max_errors = self.config.max_errors
        )
        
        if response is BrowserStatus.ERROR:

            return BrowserStatus.ERROR

        if response.status_code != 200:

            return ServerError(
                message = response.text,
                status_code = response.status_code
            )

        return BrowserStatus.SUCCESS
