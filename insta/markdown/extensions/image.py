from insta.markdown.schemas import (
    register_schema,
    remove_newlines,
    ImageSchema,
    DEFAULT_INDENT_VALUE,
    MARKDOWN_SCHEMAS,
    TYPE_TO_SCHEMA,
    clean_label,
)

from insta.markdown.build import (
    MarkdownNode
)

from insta.configs.browser_config import (
    NodeMetadata
)

from typing import List


@register_schema(
    "insta_image",
    priority = 0,
    behaves_like = [
        "image"
    ]
)
class InSTAImageSchema(ImageSchema):

    def format(
        self, node: MarkdownNode,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:

        image_title = remove_newlines(
            clean_label(node.html_element.attrib.get("alt")) or 
            clean_label(node.html_element.attrib.get("name")) or 
            clean_label(node.html_element.attrib.get("title")) or 
            clean_label(node.html_element.attrib.get("aria-label")) or 
            clean_label(node.html_element.attrib.get("src")) or ""
        )

        if len(image_title) == 0:
            return "image"

        return "{title} image".format(
            title = image_title
        )


# remove the default image schema
# this will be replaced by the InSTAImageSchema

if "image" in TYPE_TO_SCHEMA:

    MARKDOWN_SCHEMAS.pop(
        MARKDOWN_SCHEMAS.index(
            TYPE_TO_SCHEMA["image"]
        )
    )

    TYPE_TO_SCHEMA.rebuild()
