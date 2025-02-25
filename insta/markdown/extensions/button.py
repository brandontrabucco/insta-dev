from insta.markdown.schemas import (
    register_schema,
    remove_newlines,
    DEFAULT_INDENT_VALUE,
    ALL_SCHEMA_NAMES,
    EMPTY_TEXT,
    clean_label,
)

from insta.markdown.build import (
    MarkdownNode
)

from insta.markdown.extensions.base import (
    InSTABaseSchema
)

from insta.utils import (
    NodeMetadata
)

from typing import List


@register_schema(
    "insta_button",
    priority = 0,
    behaves_like = [
        "link"
    ]
)
class InSTAButtonSchema(InSTABaseSchema):

    transitions = [
        "insta_button",
        "insta_checkbox",
        "insta_image",
        "insta_input",
        "insta_link",
        "insta_range",
        "insta_select",
        "insta_textarea",
        *ALL_SCHEMA_NAMES
    ]

    tags = [
        'button'
    ]

    attributes = {"role": [
        'button'
    ]}

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:
        
        node_metadata = node_metadata or {}
        
        inner_text = clean_label(" ".join(
            child_representations
        ))

        title = (
            clean_label(node.html_element.attrib.get("title")) or 
            clean_label(node.html_element.attrib.get("aria-label")) or 
            (inner_text if inner_text not in EMPTY_TEXT else "#")
        )
        
        candidate_id = node_metadata[
            "candidate_id"
        ]

        return "[id: {id}] {title} button".format(
            id = candidate_id,
            title = title
        )