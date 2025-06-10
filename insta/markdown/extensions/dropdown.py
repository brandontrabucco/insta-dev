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


@register_schema(
    "insta_dropdown",
    priority = 9,
    behaves_like = [
        "link"
    ]
)
class InSTADropdownSchema(InSTABaseSchema):

    attributes = {"role": [
        'listbox',
        'combobox'
    ]}

    def format(
        self, node: MarkdownNode,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:

        backend_node_id = node.metadata[
            "backend_node_id"
        ]

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

        dropdown_title = (
            clean_label(node.html_element.attrib.get("name")) or 
            clean_label(node.html_element.attrib.get("title")) or 
            clean_label(node.html_element.attrib.get("aria-label")) or 
            (label if label not in EMPTY_TEXT else "")
        )

        dropdown_title_outputs = []

        if len(dropdown_title) > 0:

            dropdown_title_outputs.append(
                dropdown_title
            )

        title = " ".join(
            dropdown_title_outputs
            + ["dropdown"]
        )
        
        if len(child_representations) <= 1:
        
            inner_text = clean_label(" ".join(
                child_representations
            ))

            inner_text = (
                inner_text if inner_text not in EMPTY_TEXT
                else "None selected"
            )

            return "[id: {id}] \"{value}\" {title}".format(
                id = backend_node_id,
                value = inner_text,
                title = title
            )

        elif len(child_representations) > 1:

            inner_text = "\n".join([

                "{}* {}".format(
                    indent_value * indent_level,
                    list_item_text.strip()
                )

                for list_item_text, child_node in zip(
                    child_representations,
                    node.children
                )

            ])

            return "[id: {id}] {title}\n{value}\n".format(
                id = backend_node_id,
                value = inner_text,
                title = title
            )
