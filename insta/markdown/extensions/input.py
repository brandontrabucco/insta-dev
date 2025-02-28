from insta.markdown.schemas import (
    register_schema,
    remove_newlines,
    DEFAULT_INDENT_VALUE,
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
    "insta_input",
    priority = 1,
    behaves_like = [
        "link"
    ]
)
class InSTAInputSchema(InSTABaseSchema):

    attributes = {"role": [
        'input',
        'textfield'
    ], "type": [
        "button",
        "color",
        "date",
        "datetime-local",
        "email",
        "file",
        "hidden",
        "image",
        "month",
        "number",
        "password",
        "reset",
        "search",
        "submit",
        "tel",
        "text",
        "time",
        "url",
        "week"
    ]}

    def format(
        self, node: MarkdownNode,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:

        value = str(
            node.metadata.get("editable_value") or
            node.html_element.attrib.get("value") or 
            node.html_element.attrib.get("placeholder") or ""
        )

        title = (
            clean_label(node.html_element.attrib.get("name")) or 
            clean_label(node.html_element.attrib.get("title")) or 
            clean_label(node.html_element.attrib.get("aria-label")) or ""
        )

        title_outputs = []

        if len(title) > 0:
            title_outputs.append(title)

        title_outputs.append(
            node.html_element.attrib.get(
                "type", "text"
            )
        )

        title = " ".join(
            title_outputs
        )
    
        candidate_id = node.metadata[
            "candidate_id"
        ]

        return '[id: {id}] "{value}" ({title} input)'.format(
            id = candidate_id,
            value = value,
            title = title
        )
