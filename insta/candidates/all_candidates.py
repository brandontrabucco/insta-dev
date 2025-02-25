from insta.candidates.base_candidates import (
    BaseCandidates
)
from insta.utils import (
    BrowserObservation
)

import lxml
import lxml.html


class AllCandidates(BaseCandidates):
    """Candidate processor that marks a subset of elements on a webpage as
    candidates for an LLM agent to interact with.
    
    """

    def update(self, obs: BrowserObservation):
        """Process the latest observation and update the metadata to select
        a set of interactive elements as candidates for an LLM agent
        to interact with, helpful to limit the size of the text observation.

        Arguments:

        obs: PlaywrightObservation
            The latest observation from the Playwright agent.
        
        """

        for backend_node_id, node_metadata in obs.metadata.items():

            node_metadata["candidate_id"] = (
                backend_node_id
            )
