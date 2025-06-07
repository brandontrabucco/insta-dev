from insta.markdown.schemas import (
    MarkdownSchema,
    ALL_SCHEMA_NAMES
)

from insta.configs.browser_config import (
    NodeMetadata
)

import lxml.html


ALL_INSTA_SCHEMA_NAMES = [
    "insta_button",
    "insta_checkbox",
    "insta_form",
    "insta_image",
    "insta_input",
    "insta_link",
    "insta_range",
    "insta_select",
    "insta_textarea",
    "insta_dropdown",
]


class InSTABaseSchema(MarkdownSchema):

    transitions = [
        *ALL_INSTA_SCHEMA_NAMES,
        *ALL_SCHEMA_NAMES
    ]
    
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
            'backend_node_id' in node_metadata 
        )
