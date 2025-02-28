from insta.observation_processors.base_processor import (
    BaseProcessor
)

from insta.utils import (
    BrowserObservation,
    BrowserStatus,
    safe_call
)

from insta.markdown import (
    get_markdown_tree,
    render_markdown_tree,
    extensions
)

from typing import Tuple


FAILED_MESSAGE = "The page has failed to parse into markdown. Please check the logs for more information, and file a bug report to: https://github.com/data-for-agents/environment/issues"


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
        require_frontmost: bool = True
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
            observation.metadata,
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

        return BrowserObservation(
            raw_html = observation.raw_html,
            screenshot = observation.screenshot,
            metadata = observation.metadata,
            current_url = observation.current_url,
            processed_text = " ".join(outputs)
        )

