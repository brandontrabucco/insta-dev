from insta.markdown.schemas import (
    register_schema,
    remove_newlines,
    MarkdownSchema,
    DEFAULT_INDENT_VALUE,
    ALL_SCHEMA_NAMES,
    clean_label,
)

from insta.markdown.build import (
    MarkdownNode
)

from insta.markdown.extensions.base import (
    InSTABaseSchema,
    ALL_INSTA_SCHEMA_NAMES
)

from insta.configs.browser_config import (
    NodeMetadata
)

from typing import List


@register_schema(
    "insta_form",
    priority = 0,
    behaves_like = [
        "link"
    ]
)
class InSTAFormSchema(MarkdownSchema):

    transitions = [
        *ALL_INSTA_SCHEMA_NAMES,
        *ALL_SCHEMA_NAMES
    ]

    tags = [
        'form'
    ]
    
    def format(
        self, node: MarkdownNode,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:
        
        inner_text = "\n".join(
            child_representations
        )

        title = (
            clean_label(node.html_element.attrib.get("name")) or 
            clean_label(node.html_element.attrib.get("title")) or 
            clean_label(node.html_element.attrib.get("aria-label")) or 
            clean_label(node.html_element.attrib.get("role")) or 
            clean_label(node.html_element.attrib.get("action")) or ""
        )

        title_outputs = ["##"]

        if len(title) > 0:
            title_outputs.append(title)
        
        title_outputs.append(
            "form"
        )

        title = " ".join(
            title_outputs
        )

        return "\n{title}\n{inner_text}".format(
            title = title.title(),
            inner_text = inner_text
        )