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
    "insta_range",
    priority = 0,
    behaves_like = [
        "link"
    ]
)
class InSTARangeSchema(InSTABaseSchema):

    attributes = {"role": [
        'slider',
        'spinbutton'
    ], "type": [
        'range'
    ]}
    
    def format(
        self, node: MarkdownNode,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:

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
            "range slider"
        )

        range_min = (
            node.html_element.attrib.get("min") or
            node.html_element.attrib.get("aria-valuemin")
        )

        if range_min is not None:
            title_outputs.append(
                "min: {}".format(range_min)
            )

        range_max = (
            node.html_element.attrib.get("max") or 
            node.html_element.attrib.get("aria-valuemax")
        )

        if range_max is not None:
            title_outputs.append(
                "max: {}".format(range_max)
            )

        range_step = (
            node.html_element.attrib.get("step") or 
            node.html_element.attrib.get("aria-valuetext")
        )

        if range_step is not None:
            title_outputs.append(
                "step: {}".format(range_step)
            )

        title = " ".join(
            title_outputs
        )

        real_value = (
            node.metadata.get("editable_value") or
            node.html_element.attrib.get("aria-valuenow") or 
            node.html_element.attrib.get("value")
        )

        display_value = str(
            node.html_element.attrib.get("aria-valuetext")
        )

        backend_node_id = node.metadata[
            "backend_node_id"
        ]

        if display_value is None:

            return '[id: {id}] "{real_value}" ({title})'.format(
                id = backend_node_id,
                real_value = real_value,
                title = title
            )

        return '[id: {id}] "{display_value} ({real_value})" ({title})'.format(
            id = backend_node_id,
            display_value = display_value,
            real_value = real_value,
            title = title
        )
