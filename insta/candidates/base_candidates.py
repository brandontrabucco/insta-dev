from insta.utils import (
    BrowserObservation
)

import abc


class BaseCandidates(abc.ABC):
    """Candidate processor that marks a subset of elements on a webpage as
    candidates for an LLM agent to interact with.
    
    """

    @abc.abstractmethod
    def update(self, obs: BrowserObservation):
        """Process the latest observation and update the metadata to select
        a set of interactive elements as candidates for an LLM agent
        to interact with, helpful to limit the size of the text observation.

        Arguments:

        obs: PlaywrightObservation
            The latest observation from the Playwright agent.
        
        """

        return NotImplemented
