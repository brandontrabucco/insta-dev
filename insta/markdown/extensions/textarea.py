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
    "insta_textarea",
    priority = 0,
    behaves_like = [
        "link"
    ]
)
class InSTATextareaSchema(InSTABaseSchema):

    tags = [
        'textarea',
    ]

    attributes = {"role": [
        'textbox'
    ], "contenteditable": [
        'true'
    ]}

    def format(
        self, node: MarkdownNode,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:

        content = str(
            node.html_element.attrib.get("aria-valuetext") or 
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
            "textbox"
        )

        title = " ".join(
            title_outputs
        )
        
        candidate_id = node.metadata[
            "candidate_id"
        ]

        return '[id: {id}] """\n{content}\n""" ({title})'.format(
            id = candidate_id,
            content = content,
            title = title
        )