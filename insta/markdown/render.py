from insta.markdown.schemas import (
    TYPE_TO_SCHEMA,
    DEFAULT_INDENT_VALUE
)
from insta.markdown.build import (
    MarkdownNode
)

from insta.utils import (
    NodeToMetadata
)

from typing import List


def render_markdown_tree(
    markdown_nodes: List[MarkdownNode],
    indent_level: int = 0,
    indent_value: str = DEFAULT_INDENT_VALUE,
) -> List[str]:
    """Render a list of MarkdownNodes into actual Markdown text.

    Arguments:

    markdown_nodes: List[MarkdownNode]
        A list of MarkdownNodes representing the tree structure of the
        top level HTML elements in the input HTML string.

    indent_level: int
        The current level of indentation in the rendered text.

    indent_value: str
        The string to use to further indent the text.

    Returns:

    List[str]
        A list of strings, each representing a MarkdownNode rendered into
        a corresponding Markdown text representation.
    
    """
    
    representations = []

    for node in markdown_nodes:
        
        node_schema = TYPE_TO_SCHEMA[
            node.type
        ]

        child_representations = []

        if node.children is not None:

            next_indent_level = (
                indent_level +
                node_schema.increment_indent
            )

            child_representations = render_markdown_tree(
                markdown_nodes = node.children,
                indent_level = next_indent_level,
                indent_value = indent_value,
            )

        output_text = node_schema.format(
            node = node, child_representations = child_representations,
            indent_level = indent_level,
            indent_value = indent_value,
        )

        representations.append(output_text)
    
    return representations