from insta.markdown.schemas import (
    MarkdownSchema,
    MarkdownNode,
    MARKDOWN_SCHEMAS,
    TYPE_TO_SCHEMA,
)

from insta.utils import (
    NodeToMetadata,
    NodeMetadata
)

from typing import List, Tuple

from lxml.html import HtmlElement
import lxml.html
import lxml.html.clean


HTMLDOMNode = HtmlElement | str


def get_text_and_children(
    html_element: HtmlElement
) -> List[HTMLDOMNode]:
    
    parts = []

    has_start_text = (
        html_element.text is not None
        and len(html_element.text.strip()) > 0
    )

    if has_start_text:

        parts.append(
            html_element.text.strip()
        )

    for child in html_element.getchildren():

        parts.append(child)

        has_end_text = (
            child.tail is not None
            and len(child.tail.strip()) > 0
        )

        if has_end_text:

            parts.append(
                child.tail.strip()
            )
            
    return parts


def match_schema(
    schema: MarkdownSchema,
    html_element: HtmlElement,
    node_metadata: NodeMetadata = None,
    metadata: NodeToMetadata = None,
    last_markdown_node: MarkdownNode = None,
    last_html_node: HtmlElement = None,
    restrict_viewport: Tuple[float, float, float, float] = None,
    require_visible: bool = True,
    require_frontmost: bool = True,
) -> bool:
    
    if last_markdown_node is not None:

        last_markdown_schema = TYPE_TO_SCHEMA[
            last_markdown_node.type
        ]

        transition_not_allowed = (
            schema.type not in 
            (last_markdown_schema.transitions or [])
        )

        if transition_not_allowed:
    
            return False
    
    if isinstance(html_element, str):  # text nodes

        node_has_metadata = (
            isinstance(last_html_node, HtmlElement) and
            metadata is not None and
            'backend_node_id' in last_html_node.attrib
        )

        if node_has_metadata:

            backend_node_id = str(
                last_html_node
                .attrib['backend_node_id']
            )

            node_metadata = metadata[
                backend_node_id
            ]

    if node_metadata is not None and \
            (require_visible or require_frontmost):

        is_visible = element_is_visible(
            metadata = node_metadata,
            require_visible = require_visible,
            require_frontmost = require_frontmost
        )

        if not is_visible:

            return False

    if node_metadata is not None and \
            restrict_viewport is not None:

        within_viewport = element_within_viewport(
            metadata = node_metadata,
            restrict_viewport = restrict_viewport
        )

        if not within_viewport:

            return False

    return schema.match(
        html_element = html_element,
        node_metadata = node_metadata
    )
    

def parse_from_schema(
    schema: MarkdownSchema,
    html_element: HtmlElement,
    node_metadata: NodeMetadata,
    metadata: NodeToMetadata,
    restrict_viewport: Tuple[float, float, float, float] = None,
    require_visible: bool = True,
    require_frontmost: bool = True,
) -> MarkdownNode:

    element_type = schema.type

    last_markdown_node = MarkdownNode(
        html_element = html_element,
        metadata = node_metadata,
        children = [],
        type = element_type,
    )

    last_html_node = html_element

    for child in get_text_and_children(html_element):

        last_markdown_node.children.extend(
            expand_markdown_tree(
                child, metadata = metadata,
                restrict_viewport = restrict_viewport,
                last_markdown_node = last_markdown_node,
                last_html_node = last_html_node,
                require_visible = require_visible,
                require_frontmost = require_frontmost
            )
        )

        if isinstance(child, HtmlElement):

            last_html_node = child

    return last_markdown_node


def element_is_visible(
    metadata: NodeMetadata,
    require_visible: bool = True,
    require_frontmost: bool = True
) -> bool:

    pass_through = (
        metadata['computed_style']['display'] 
        == 'contents'
    )

    is_visible = (
        (metadata['is_visible'] or not require_visible)
        and (metadata['is_frontmost'] or not require_frontmost)
    )

    return pass_through or is_visible


def element_within_viewport(
    metadata: NodeMetadata,
    restrict_viewport: Tuple[float, float, float, float]
) -> bool:

    bounding_client_rect = metadata[
        'bounding_client_rect'
    ]
        
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
    html_dom_node: HTMLDOMNode,
    metadata: NodeToMetadata,
    last_markdown_node: MarkdownNode = None,
    last_html_node: HtmlElement = None,
    restrict_viewport: Tuple[float, float, float, float] = None,
    require_visible: bool = True,
    require_frontmost: bool = True,
) -> List[MarkdownNode]:

    node_metadata = None

    node_has_metadata = (
        isinstance(html_dom_node, HtmlElement) and
        metadata is not None and
        'backend_node_id' in html_dom_node.attrib
    )

    if node_has_metadata:

        backend_node_id = str(
            html_dom_node.attrib[
                'backend_node_id'
            ]
        )

        node_metadata = metadata[
            backend_node_id
        ]
    
    if isinstance(html_dom_node, str) and match_schema(
        TYPE_TO_SCHEMA['text'], html_dom_node,
        node_metadata = node_metadata,
        metadata = metadata,
        last_markdown_node = last_markdown_node,
        last_html_node = last_html_node,
        restrict_viewport = restrict_viewport,
        require_visible = require_visible,
        require_frontmost = require_frontmost
    ):
        
        return [
            MarkdownNode(
                text_content = html_dom_node,
                type = 'text'
            )
        ]
    
    elif isinstance(html_dom_node, str):

        return []
    
    output_nodes = []

    for schema in MARKDOWN_SCHEMAS:

        if match_schema(
            schema, html_dom_node,
            node_metadata = node_metadata,
            metadata = metadata,
            last_markdown_node = last_markdown_node,
            last_html_node = last_html_node,
            restrict_viewport = restrict_viewport,
            require_visible = require_visible,
            require_frontmost = require_frontmost
        ):

            output_nodes.append(
                parse_from_schema(
                    schema, html_dom_node,
                    node_metadata = node_metadata,
                    metadata = metadata,
                    restrict_viewport = restrict_viewport,
                    require_visible = require_visible,
                    require_frontmost = require_frontmost
                )
            )

            break 

    if len(output_nodes) > 0:

        return output_nodes

    last_html_node = html_dom_node
    
    for child in get_text_and_children(html_dom_node):

        output_nodes.extend(
            expand_markdown_tree(
                child, metadata = metadata,
                restrict_viewport = restrict_viewport,
                last_markdown_node = last_markdown_node,
                last_html_node = last_html_node,
                require_visible = require_visible,
                require_frontmost = require_frontmost
            )
        )

        if isinstance(child, HtmlElement):

            last_html_node = child

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
    restrict_viewport: Tuple[float, float, float, float] = None,
    require_visible: bool = True,
    require_frontmost: bool = True,
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

    require_visible: bool
        Boolean flag indicating whether the observation should only include
        elements that are current in a visible state.

    require_frontmost: bool
        Boolean flag indicating whether the observation should only include
        elements that are currently in the frontmost layer.

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
        restrict_viewport = restrict_viewport,
        require_visible = require_visible,
        require_frontmost = require_frontmost
    )