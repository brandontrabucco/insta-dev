from insta.observation_processors.base_processor import (
    BaseProcessor
)

from insta.observation_processors.pii_tools import (
    scrubadub_clean
)

from insta.utils import (
    BrowserStatus,
    safe_call
)

from insta.configs.browser_config import (
    BrowserObservation
)

from insta.markdown import (
    get_markdown_tree,
    render_markdown_tree,
    extensions
)

from typing import Tuple


FAILED_MESSAGE = "The page has failed to parse into markdown. Please check the logs for more information, and file a bug report to: https://github.com/data-for-agents/insta/issues"


CATCH_PARSE_ERRORS = True
LOG_PARSE_ERRORS = True
MAX_PARSE_ERRORS = 1


class MarkdownProcessor(BaseProcessor):
    """Observation processor that converts raw HTML into an agent-readable format,
    and optionally restricts the observation to the current viewport.    
    
    """

    def process(
        self, observation: BrowserObservation,
        restrict_viewport: Tuple[float, float, float, float] = None,
        require_visible: bool = True,
        require_frontmost: bool = True,
        remove_pii: bool = True
    ) -> BrowserObservation:
        """Process the latest observation from a web browsing environment, 
        and create an agent-readible observation, with an option to
        restrict the observation to the current viewport.

        Arguments:

        observation: PlaywrightObservation
            The latest observation from the web browsing environment.

        restrict_viewport: Tuple[float, float, float, float]
            A tuple of the form (x, y, width, height) that restricts the 
            observation to the current viewport.

        require_visible: bool
            Boolean indicating whether the observation should only include
            elements that are current in a visible state.

        require_frontmost: bool
            Boolean indicating whether the observation should only include
            elements that are currently in the frontmost layer.

        remove_pii: bool
            Boolean indicating whether the observation should remove any
            personally identifiable information.

        Returns:

        observation: PlaywrightObservation
            An updated observation with a `processed_text` field that contains
            an agent-readable version of the observation.
        
        """

        markdown_nodes = safe_call(
            get_markdown_tree,
            observation.raw_html,
            observation.metadata,
            restrict_viewport = restrict_viewport,
            require_visible = require_visible,
            require_frontmost = require_frontmost,
            catch_errors = CATCH_PARSE_ERRORS,
            log_errors = LOG_PARSE_ERRORS,
            max_errors = MAX_PARSE_ERRORS
        )

        if markdown_nodes is BrowserStatus.ERROR:

            return BrowserObservation(
                raw_html = observation.raw_html,
                screenshot = observation.screenshot,
                metadata = observation.metadata,
                current_url = observation.current_url,
                processed_text = FAILED_MESSAGE
            )

        outputs = safe_call(
            render_markdown_tree,
            markdown_nodes,
            catch_errors = CATCH_PARSE_ERRORS,
            log_errors = LOG_PARSE_ERRORS,
            max_errors = MAX_PARSE_ERRORS
        )

        if outputs is BrowserStatus.ERROR:

            return BrowserObservation(
                raw_html = observation.raw_html,
                screenshot = observation.screenshot,
                metadata = observation.metadata,
                current_url = observation.current_url,
                processed_text = FAILED_MESSAGE
            )
        
        processed_text = " ".join(outputs)

        if remove_pii:  # remove PII using the scrubadub library

            processed_text = scrubadub_clean(
                processed_text
            ) 

        return BrowserObservation(
            raw_html = observation.raw_html,
            screenshot = observation.screenshot,
            metadata = observation.metadata,
            current_url = observation.current_url,
            processed_text = processed_text
        )

