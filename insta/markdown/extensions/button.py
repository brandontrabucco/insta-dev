from insta.markdown.schemas import (
    register_schema,
    remove_newlines,
    DEFAULT_INDENT_VALUE,
    EMPTY_TEXT,
    clean_label,
)

from insta.markdown.build import (
    MarkdownNode
)

from insta.markdown.extensions.base import (
    InSTABaseSchema
)

from insta.configs.browser_config import (
    NodeMetadata
)

from typing import List


DEFAULT_TITLE = "#"


@register_schema(
    "insta_button",
    priority = 0,
    behaves_like = [
        "link"
    ]
)
class InSTAButtonSchema(InSTABaseSchema):

    tags = [
        'button'
    ]

    attributes = {"role": [
        'button',
        'option'
    ]}

    def format(
        self, node: MarkdownNode,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:
        
        inner_text = clean_label(" ".join(
            child_representations
        ))

        backend_node_id = node.metadata[
            "backend_node_id"
        ]

        button_title = (
            clean_label(node.html_element.attrib.get("name")) or 
            clean_label(node.html_element.attrib.get("title")) or 
            clean_label(node.html_element.attrib.get("aria-label")) or 
            (inner_text if inner_text not in EMPTY_TEXT else "")
        )

        button_type = node.html_element.attrib.get(
            "type", None
        )

        button_role = node.html_element.attrib.get(
            "role", 'button'
        )

        button_title_outputs = []

        if len(button_title) > 0:

            button_title_outputs.append(
                button_title
            )

        if button_type is not None and button_type != "button":

            button_title_outputs.append(
                button_type
            )

        title = " ".join(
            button_title_outputs
        ) or DEFAULT_TITLE

        return "[id: {id}] {title} {role}".format(
            id = backend_node_id,
            title = title,
            role = button_role
        )
