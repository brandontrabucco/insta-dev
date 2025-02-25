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
    metadata: NodeToMetadata,
    indent_level: int = 0,
    indent_value: str = DEFAULT_INDENT_VALUE,
) -> List[str]:
    """Render a list of MarkdownNodes into actual Markdown text.

    Arguments:

    markdown_nodes: List[MarkdownNode]
        A list of MarkdownNodes representing the tree structure of the
        top level HTML elements in the input HTML string.

    metadata: NodeToMetadata
        A dictionary mapping backend node IDs to metadata, including the
        bounding_client_rect of the corresponding HTML element, the
        computed_style of the corresponding HTML element, and 
        other useful metadata extracted from the DOM.

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
        
        node_metadata = None

        node_has_metadata = (
            node.html_element is not None and
            'backend_node_id' in node.html_element.attrib
        )

        if node_has_metadata:

            backend_node_id = str(node.html_element.attrib[
                'backend_node_id'
            ])

            node_metadata = metadata[
                backend_node_id
            ]

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
                node.children,
                metadata = metadata,
                indent_level = next_indent_level,
                indent_value = indent_value,
            )

        output_text = node_schema.format(
            node = node, node_metadata = node_metadata,
            child_representations = child_representations,
            indent_level = indent_level,
            indent_value = indent_value,
        )

        representations.append(output_text)
    
    return representations