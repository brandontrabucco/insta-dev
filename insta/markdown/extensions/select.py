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
    "insta_select",
    priority = 0,
    behaves_like = [
        "link"
    ]
)
class InSTASelectSchema(InSTABaseSchema):

    tags = [
        'select'
    ]
    
    def format(
        self, node: MarkdownNode,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:

        options = node.html_element.findall(
            ".//option"
        )

        options_labels = [
            option.text or option.attrib.get("value") or ""
            for option in options
        ]
        
        editable_value = node.metadata.get(
            "editable_value"
        )

        selected_label = ""

        valid_editable_value = (
            editable_value is not None and 
            isinstance(editable_value, int) and 
            0 <= editable_value < len(options_labels)
        )

        if valid_editable_value:

            selected_label = options_labels[
                editable_value
            ]

        options_text = ", ".join(
            options_labels
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
            "select"
        )

        title = " ".join(
            title_outputs
        )

        candidate_id = node.metadata[
            "candidate_id"
        ]

        return '[id: {id}] "{value}" ({title} from: {options})'.format(
            id = candidate_id,
            value = selected_label,
            title = title,
            options = options_text
        )
