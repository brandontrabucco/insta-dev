import scrubadub
import scrubadub_spacy


SCRUBBER: scrubadub.Scrubber = None


def scrubadub_clean(processed_text: str) -> str:
    """Remove all personally identifiable information from text.

    Arguments:

    processed_text: str
        Input text to remove PII from.

    Returns:

    processed_text: str
        The text with PII removed.
    
    """

    global SCRUBBER

    if SCRUBBER is None:

        SCRUBBER = scrubadub.Scrubber()

        # detect names and entities using spacy
        SCRUBBER.add_detector(
            scrubadub_spacy.detectors.SpacyEntityDetector
        )

    return SCRUBBER.clean(
        processed_text
    )
