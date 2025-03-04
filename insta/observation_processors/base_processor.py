from insta.utils import (
    BrowserObservation
)

from typing import Tuple

import abc

class BaseProcessor(abc.ABC):
    """Observation processor that converts raw HTML into an agent-readable format,
    and optionally restricts the observation to the current viewport.    
    
    """

    @abc.abstractmethod
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

        return NotImplemented