from insta.markdown.schemas import (
    MarkdownSchema,
    MarkdownNode,
    MARKDOWN_SCHEMAS,
    TYPE_TO_SCHEMA,
)

from insta.utils import (
    NodeToMetadata,
    NodeMetadata,
    SKIP_TAGS
)

from typing import List, Tuple

import lxml.html
import lxml.html.clean


HTMLDOMNode = lxml.html.HtmlElement | str


def get_text_and_children(
    html_element: lxml.html.HtmlElement
) -> List[HTMLDOMNode]:
    
    parts = []

    if html_element.text is not None \
            and len(html_element.text.strip()) > 0:

        parts.append(
            html_element.text.strip()
        )

    for child in html_element.getchildren():

        parts.append(child)

        if child.tail is not None \
                and len(child.tail.strip()) > 0:

            parts.append(
                child.tail.strip()
            )
            
    return parts


def match_schema(
    schema: MarkdownSchema,
    html_element: lxml.html.HtmlElement,
    node_metadata: NodeMetadata = None,
    last_node: MarkdownNode = None
) -> bool:
    
    if last_node is not None:

        last_schema = TYPE_TO_SCHEMA[
            last_node.type
        ]

        transition_not_allowed = (
            schema.type not in 
            (last_schema.transitions or [])
        )

        if transition_not_allowed:
            return False

    return schema.match(
        html_element = html_element,
        node_metadata = node_metadata
    )
    

def parse_from_schema(
    schema: MarkdownSchema,
    html_element: lxml.html.HtmlElement,
    metadata: NodeToMetadata = None,
    restrict_viewport: Tuple[float, float, float, float] = None
) -> MarkdownNode:

    element_type = schema.type

    markdown_node = MarkdownNode(
        html_element = html_element,
        children = [],
        type = element_type,
    )

    for child in get_text_and_children(html_element):

        markdown_node.children.extend(
            expand_markdown_tree(
                child, metadata = metadata,
                restrict_viewport = restrict_viewport,
                last_node = markdown_node
            )
        )

    return markdown_node


def element_is_visible(
    computed_style: dict,
    bounding_client_rect: dict,
    html_element: lxml.html.HtmlElement
) -> bool:

    is_visible = (
        computed_style['visibility'] != 'hidden' and
        computed_style['display'] != 'none' and
        html_element.attrib.get('aria-hidden') != 'true'
    )

    return is_visible


def element_within_viewport(
    bounding_client_rect: dict,
    restrict_viewport: Tuple[float, float, float, float]
) -> bool:
        
    elem_x0 = bounding_client_rect['x']
    elem_y0 = bounding_client_rect['y']

    elem_width = bounding_client_rect['width']
    elem_height = bounding_client_rect['height']

    elem_x1 = elem_x0 + elem_width
    elem_y1 = elem_y0 + elem_height

    view_x0, view_y0, view_width, view_height = restrict_viewport

    view_x1 = view_x0 + view_width
    view_y1 = view_y0 + view_height
    
    x_overlap = (
        elem_x0 <= view_x1 and
        view_x0 <= elem_x1
    )

    y_overlap = (
        elem_y0 <= view_y1 and
        view_y0 <= elem_y1
    )

    within_viewport = x_overlap and y_overlap

    return within_viewport


def expand_markdown_tree(
    node: HTMLDOMNode,
    metadata: NodeToMetadata = None,
    restrict_viewport: Tuple[float, float, float, float] = None,
    last_node: MarkdownNode = None
) -> List[MarkdownNode]:
    
    if isinstance(node, str):
        
        return [
            MarkdownNode(
                text_content = node,
                type = 'text'
            )
        ]

    # check if node is currently visible and not hidden
    # skip rendering if not visible

    node_metadata = None

    node_has_metadata = (
        metadata is not None and
        'backend_node_id' in node.attrib
    )

    if node_has_metadata:

        backend_node_id = str(node.attrib[
            'backend_node_id'
        ])

        node_metadata = metadata[
            backend_node_id
        ]

        bounding_client_rect = node_metadata[
            'bounding_client_rect'
        ]
        computed_style = node_metadata[
            'computed_style'
        ]

        is_visible = element_is_visible(
            computed_style,
            bounding_client_rect,
            html_element = node
        )

        if not is_visible:
            return []

        if restrict_viewport is not None:

            within_viewport = element_within_viewport(
                bounding_client_rect,
                restrict_viewport
            )

            if not within_viewport:
                return []
    
    output_nodes = []

    for schema in MARKDOWN_SCHEMAS:

        if match_schema(
            schema, node,
            node_metadata = node_metadata,
            last_node = last_node
        ):

            output_nodes.append(
                parse_from_schema(
                    schema, node,
                    metadata = metadata,
                    restrict_viewport = restrict_viewport
                )
            )

            break 

    text_and_children = get_text_and_children(node)

    if len(output_nodes) == 0 \
            and len(text_and_children) > 0:
        
        for child in text_and_children:

            output_nodes.extend(
                expand_markdown_tree(
                    child, metadata = metadata,
                    restrict_viewport = restrict_viewport,
                    last_node = last_node
                )
            )

    return output_nodes


CLEANER = lxml.html.clean.Cleaner(
    scripts = True,
    javascript = True,
    comments = True,
    style = True,
    links = True,
    meta = True,
    page_structure = False,
    processing_instructions = True,
    embedded = False,
    frames = False,
    forms = False,
    annoying_tags = True,
    remove_unknown_tags = True,
    safe_attrs_only = False,
    add_nofollow = False,
    remove_tags = [
        'noscript'
    ],
    kill_tags = [
        'noscript'
    ]
)


def get_markdown_tree(
    raw_html: str,
    metadata: NodeToMetadata,
    restrict_viewport: Tuple[float, float, float, float] = None
) -> List[MarkdownNode]:
    """Process an HTML string into a tree of MarkdownNodes, an intermediate step
    before rendering the markdown tree into a string.

    Arguments:

    raw_html: str
        The HTML string to be processed.

    metadata: NodeToMetadata
        A dictionary mapping backend node IDs to metadata, including the
        bounding_client_rect of the corresponding HTML element, the
        computed_style of the corresponding HTML element, and 
        other useful metadata extracted from the DOM.

    restrict_viewport: Tuple[float, float, float, float]
        A tuple of the form (x, y, width, height) that restricts the 
        observation to the current viewport.

    Returns:

    List[MarkdownNode]
        A list of MarkdownNodes representing the tree structure of the
        top level HTML elements in the input HTML string.
    
    """

    root_node = CLEANER.clean_html(
        lxml.html.fromstring(
            raw_html
        )
    )

    return expand_markdown_tree(
        root_node, metadata = metadata,
        restrict_viewport = restrict_viewport
    )