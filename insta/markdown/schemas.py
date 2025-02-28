from insta.utils import (
    NodeMetadata
)

from typing import List, Dict, Any
from dataclasses import dataclass, field

import lxml
import lxml.html
import re


EMPTY_TEXT = [
    "",
    "#",
    "##",
    "**",
    "__",
    "~~",
    "``",
    "''",
    '""',
    "[]",
    "{}",
    "()",
    "<>",
    "****",
    "____",
    "~~~~",
]


MAX_LABEL_LENGTH = 64


def remove_newlines(text: str) -> str:

    return re.sub(
        r"\n+",
        " ",
        text
    ).strip()


def clean_label(
    label: str | None
) -> str | None:
    
    if label is None:
        return None
    
    label = label.strip()

    if label in EMPTY_TEXT:
        return ""
    
    if len(label) > MAX_LABEL_LENGTH:

        label = label[
            :MAX_LABEL_LENGTH
        ] + "..."
    
    return label


MarkdownNodeType = str


@dataclass
class MarkdownNode:
    
    html_element: lxml.html.HtmlElement = None
    text_content: str = None

    metadata: NodeMetadata = None

    children: List['MarkdownNode'] = field(
        default_factory = list
    )

    type: MarkdownNodeType = None


DEFAULT_INDENT_VALUE = "    "


class MarkdownSchema:

    increment_indent: int = 0
    
    type: MarkdownNodeType = None
    transitions: List[MarkdownNodeType] = None

    tags: List[str] = None
    attributes: Dict[str, Any] = None
    
    def match(
        self, html_element: lxml.html.HtmlElement,
        node_metadata: NodeMetadata = None,
    ) -> bool:
        
        valid_attributes = (
            self.attributes or {}
        )

        valid_tags  = (
            self.tags or []
        )

        match_attributes = any([
            html_element.get(key) in values
            for key, values in valid_attributes.items()
        ])

        match_tag = html_element.tag in valid_tags

        return (
            match_attributes or
            match_tag
        )

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:

        return NotImplemented


MARKDOWN_SCHEMAS: List[MarkdownSchema] = []


def register_schema(
    name: str,
    priority: int = None,
    behaves_like: List[str] = None
):

    for other in MARKDOWN_SCHEMAS:

        if other.transitions is None:
            continue

        for transition in behaves_like or []:

            if transition in other.transitions \
                    and name not in other.transitions:

                other.transitions.append(name)
    
    def inner(
        cls: MarkdownSchema
    ):

        schema = cls()
        schema.type = name

        MARKDOWN_SCHEMAS.insert(
            priority if priority is not None else 
            len(MARKDOWN_SCHEMAS),
            schema
        )

        return cls
    
    return inner


@register_schema('ordered_list')
class OrderedListSchema(MarkdownSchema):

    increment_indent = 1

    transitions = [
        'list_item',
        'link',
        'text',
        'bold',
        'italic',
        'underline',
        'strikethrough',
        'inline_code',
        'image',
    ]

    tags = [
        'ol'
    ]

    attributes = {"role": [
        'list',
        'menu',
        'menubar',
        'radiogroup',
        'tablist',
        'tree'
    ]}

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:
        
        output_text = "\n".join([

            "{}{} {}".format(
                indent_value * indent_level,
                "" if child_node.type != 'list_item' else
                "{idx}.".format(idx = idx + 1),
                list_item_text.strip()
            )

            for idx, (list_item_text, child_node) in enumerate(zip(
                child_representations, node.children
            ))

        ])

        return "\n" + output_text + "\n"


@register_schema('unordered_list')
class UnorderedListSchema(MarkdownSchema):

    increment_indent = 1

    transitions = [
        'list_item',
        'link',
        'text',
        'bold',
        'italic',
        'underline',
        'strikethrough',
        'inline_code',
        'image',
    ]

    tags = [
        'ul'
    ]

    attributes = {"role": [
        'list',
        'menu',
        'menubar',
        'radiogroup',
        'tablist',
        'tree'
    ]}

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:
        
        output_text = "\n".join([

            "{}{} {}".format(
                indent_value * indent_level,
                "" if child_node.type != 'list_item' else "*",
                list_item_text.strip()
            )

            for list_item_text, child_node in zip(
                child_representations,
                node.children
            )

        ])

        return "\n" + output_text + "\n"


@register_schema('list_item')
class ListItemSchema(MarkdownSchema):

    transitions = [
        'unordered_list',
        'ordered_list',
        'heading',
        'link',
        'text',
        'bold',
        'italic',
        'underline',
        'strikethrough',
        'inline_code',
        'image',
    ]

    tags = [
        'li'
    ]

    attributes = {"role": [
        "listitem",
        "menuitem",
        "radio",
        "tab",
        "treeitem"
    ]}

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:

        output_text = " ".join(child_representations)

        return output_text


@register_schema('table')
class TableSchema(MarkdownSchema):

    transitions = [
        'table_row'
    ]

    tags = [
        'table'
    ]

    attributes = {"role": [
        'table',
        'grid',
        'treegrid'
    ]}

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:
        
        if len(child_representations) == 0:

            return ""

        max_cells = max([
            len(row.split("|"))
            for row in child_representations
        ])

        output_rows = []

        for idx, row in enumerate(child_representations):

            num_cells_in_row = len(row.split("|"))

            num_cells_to_add = (
                max_cells - 
                num_cells_in_row
            )

            if num_cells_to_add > 0:

                row = row + " | " * num_cells_to_add
            
            output_rows.append(row)

            if idx == 0:

                output_rows.append(" --- ".join(
                    ["|"] * (max_cells - 1)
                ))

        output_text = (
            "\n" + "\n".join(output_rows) + "\n"
        )

        return output_text


@register_schema('table_row')
class TableRowSchema(MarkdownSchema):

    transitions = [
        'table_cell'
    ]

    attributes = {"role": [
        'row',
    ]}

    tags = [
        'tr'
    ]

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:

        output_cols = []

        for col_idx, col in enumerate(
            child_representations
        ):

            col = remove_newlines(
                col).strip()

            colspan = None

            has_html_element = (
                col_idx < len(node.children) and 
                node.children[col_idx].html_element is not None
            )

            if has_html_element:

                colspan_str = (
                    node.children[col_idx].html_element
                    .get('colspan', '1')
                )

                is_numeric = all([
                    x.isdigit() for x in colspan_str
                ]) and len(colspan_str) > 0
                
                if is_numeric:
                    
                    colspan = int(colspan_str)

            if colspan is not None and colspan > 1:

                col += " | " * (colspan - 1)

            output_cols.append(col)

        output_text = (
            "| " + " | ".join(output_cols) + " |"
        )

        return output_text


@register_schema('table_cell')
class TableCellSchema(MarkdownSchema):

    transitions = [
        'heading',
        'link',
        'text',
        'bold',
        'italic',
        'underline',
        'strikethrough',
        'inline_code',
        'image',
        'ordered_list',
        'unordered_list',
    ]

    tags = [
        'th',
        'td'
    ]

    attributes = {"role": [
        'rowheader',
        'gridcell',
        'columnheader',
    ]}

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:

        output_text = " ".join(child_representations)

        return remove_newlines(output_text)


@register_schema('heading')
class HeadingSchema(MarkdownSchema):

    transitions = [
        'text',
        'bold',
        'italic',
        'underline',
        'strikethrough',
        'inline_code',
        'image',
    ]

    tags = [
        'h1',
        'h2',
        'h3',
        'h4',
        'h5',
        'h6'
    ]

    attributes = {"role": [
        'heading',
        'tab'
    ]}

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:

        heading_level = {
            "h1": 1, "h2": 2, "h3": 3,
            "h4": 4, "h5": 5, "h6": 6
        }.get(node.html_element.tag, 1)
        
        output_text = "\n{} {}\n".format(
            "#" * heading_level,
            " ".join(child_representations)
        )

        return output_text


@register_schema('image')
class ImageSchema(MarkdownSchema):

    tags = [
        'img'
    ]

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:

        output_text = "![{}]({})".format(
            node.html_element.get('alt'),
            node.html_element.get('src')
        )
        
        if node_metadata is not None:

            computed_style = node_metadata[
                "computed_style"
            ]

            if computed_style is not None and \
                    computed_style['display'] == 'block':

                output_text = (
                    output_text + "\n"
                )

        return output_text


@register_schema('link')
class LinkSchema(MarkdownSchema):

    transitions = [
        'text',
        'bold',
        'italic',
        'underline',
        'strikethrough',
        'inline_code',
    ]

    tags = [
        'a'
    ]

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:
 
        output_text = "[{}]({})".format(
            " ".join(child_representations),
            clean_label(node.html_element.get('href'))
        )
        
        if node_metadata is not None:

            computed_style = node_metadata[
                "computed_style"
            ]

            if computed_style is not None and \
                    computed_style['display'] == 'block':

                output_text = (
                    output_text + "\n"
                )

        return output_text


@register_schema('bold')
class BoldSchema(MarkdownSchema):

    transitions = [
        'text',
        'bold',
        'italic',
        'underline',
        'strikethrough',
        'inline_code',
    ]

    tags = [
        'b',
        'strong'
    ]

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:

        output_text = "**{}**".format(
            " ".join(child_representations)
        )

        return output_text


@register_schema('italic')
class ItalicSchema(MarkdownSchema):

    transitions = [
        'text',
        'bold',
        'italic',
        'underline',
        'strikethrough',
        'inline_code',
    ]

    tags = [
        'i',
        'em'
    ]

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:

        output_text = "*{}*".format(
            " ".join(child_representations)
        )

        # catch icons and emojis rendered via css
        # almost always an empty <i> tag

        if len(output_text) == 2:

            return "(icon)"

        return output_text


@register_schema('underline')
class UnderlineSchema(MarkdownSchema):

    transitions = [
        'text',
        'bold',
        'italic',
        'underline',
        'strikethrough',
        'inline_code',
    ]

    tags = [
        'u'
    ]

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:

        output_text = "__{}__".format(
            " ".join(child_representations)
        )

        return output_text


@register_schema('strikethrough')
class StrikethroughSchema(MarkdownSchema):

    transitions = [
        'text',
        'bold',
        'italic',
        'underline',
        'strikethrough',
        'inline_code',
    ]

    tags = [
        's',
        'strike',
        'del'
    ]

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:

        output_text = "~~{}~~".format(
            " ".join(child_representations)
        )

        return output_text


@register_schema('inline_code')
class InlineCodeSchema(MarkdownSchema):

    tags = [
        'code',
        'pre'
    ]

    def match(
        self, html_element: lxml.html.HtmlElement,
        node_metadata: NodeMetadata = None,
    ) -> bool:
        
        is_match = super(
            InlineCodeSchema, self
        ).match(
            html_element = html_element,
            node_metadata = node_metadata
        )

        num_lines_code = len(
            html_element.text_content()
            .split("\n")
        )

        return (
            is_match and 
            num_lines_code == 1
        )

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:

        output_text = "`{}`".format(
            " ".join(child_representations)
        )

        return output_text


@register_schema('fenced_code')
class FencedCodeSchema(MarkdownSchema):

    tags = [
        'code',
        'pre'
    ]

    def match(
        self, html_element: lxml.html.HtmlElement,
        node_metadata: NodeMetadata = None,
    ) -> bool:
        
        is_match = super(
            FencedCodeSchema, self
        ).match(
            html_element = html_element,
            node_metadata = node_metadata
        )

        num_lines_code = len(
            html_element.text_content()
            .split("\n")
        )

        return (
            is_match and 
            num_lines_code > 1
        )

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:
            
        output_text = "```\n{}\n```".format(
            " ".join(child_representations)
        )

        return "\n" + output_text + "\n"


@register_schema('blockquote')
class BlockquoteSchema(MarkdownSchema):

    transitions = [
        'text',
        'bold',
        'italic',
        'underline',
        'strikethrough',
        'inline_code',
    ]

    tags = [
        'blockquote'
    ]

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:
            
        output_text = "> {}".format(
            " ".join(child_representations)
        )

        return "\n" + remove_newlines(output_text) + "\n"


@register_schema('horizontal_rule')
class HorizontalRuleSchema(MarkdownSchema):
    
    tags = [
        'hr'
    ]

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:
            
        output_text = "\n---\n"

        return output_text


@register_schema('text')
class TextSchema(MarkdownSchema):
    
    def match(
        self, html_element: lxml.html.HtmlElement | str,
        node_metadata: NodeMetadata = None,
    ) -> bool:
        
        return isinstance(
            html_element, str
        )

    def format(
        self, node: MarkdownNode,
        node_metadata: NodeMetadata,
        child_representations: List[str],
        indent_level: int = 0,
        indent_value: str = DEFAULT_INDENT_VALUE,
    ) -> str:

        return node.text_content


ALL_SCHEMA_NAMES = [
    x.type for x in MARKDOWN_SCHEMAS
]


class SchemaLookup(dict):

    def __init__(self, *args, **kwargs):

        super(SchemaLookup, self).__init__(
            *args, **kwargs
        )

    def rebuild(self):

        self.clear()

        for x in MARKDOWN_SCHEMAS:
            self[x.type] = x

    def __getitem__(self, key):

        current_keys = sorted(list(self.keys()))
        current_schemas = sorted([
            x.type for x in MARKDOWN_SCHEMAS
        ])

        parity = (
            len(current_keys) == len(current_schemas) and all([
                x == y for x, y in zip(
                    current_keys, current_schemas
                )
            ])
        )

        if not parity:
            self.rebuild()

        return super(
            SchemaLookup, self
        ).__getitem__(key)


TYPE_TO_SCHEMA = SchemaLookup()
