from insta.markdown.schemas import (
    MarkdownSchema
)

from insta.utils import (
    NodeMetadata
)

import lxml.html


class InSTABaseSchema(MarkdownSchema):
    
    def match(
        self, html_element: lxml.html.HtmlElement,
        node_metadata: NodeMetadata = None,
    ) -> bool:
        
        is_match = super(
            InSTABaseSchema, self
        ).match(
            html_element = html_element,
            node_metadata = node_metadata
        )

        return (
            is_match and node_metadata is not None and 
            'backend_node_id' in node_metadata and
            'candidate_id' in node_metadata
        )
