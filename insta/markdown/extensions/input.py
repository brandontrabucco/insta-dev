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


BUTTON_TYPES = [
    "button",
    "reset",
    "submit",
    "image"
]

DEFAULT_TITLE = "#"


@register_schema(
    "insta_input",
    priority = 4,
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
        
        inner_text = clean_label(" ".join(
            child_representations
        ))

        value = str(
            node.metadata.get("editable_value") or
            node.html_element.attrib.get("value") or 
            node.html_element.attrib.get("placeholder") or ""
        )

        labeled_by = node.html_element.attrib.get(
            "aria-labelledby"
        )

        label = node.html_element.getroottree().find(
            ".//*[@id='{}']".format(
                labeled_by
            )
        )

        label = "" if label is None else "".join(
            label.itertext()
        ) 

        title = (
            clean_label(node.html_element.attrib.get("name")) or 
            clean_label(node.html_element.attrib.get("title")) or 
            clean_label(node.html_element.attrib.get("aria-label")) or 
            (label if label not in EMPTY_TEXT else "")
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
    
        backend_node_id = node.metadata[
            "backend_node_id"
        ]

        input_type = node.html_element.attrib.get(
            "type", None
        )

        is_button_input = (
            node.html_element.tag == "button" or 
            input_type in BUTTON_TYPES
        )

        if is_button_input:

            button_title = (
                clean_label(node.html_element.attrib.get("name")) or 
                clean_label(node.html_element.attrib.get("title")) or 
                clean_label(node.html_element.attrib.get("aria-label")) or 
                (inner_text if inner_text not in EMPTY_TEXT else "")
            )

            button_title_outputs = []

            if len(button_title) > 0:

                button_title_outputs.append(
                    button_title
                )

            if input_type is not None and input_type != "button":

                button_title_outputs.append(
                    input_type
                )

            title = " ".join(
                button_title_outputs
            ) or DEFAULT_TITLE

            return "[id: {id}] {title} button".format(
                id = backend_node_id,
                title = title
            )

        return '[id: {id}] "{value}" ({title} input)'.format(
            id = backend_node_id,
            value = value,
            title = title
        )
