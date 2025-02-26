from insta.tools.core import (
    InstaToolOutput,
    interact_with_browser
)

from insta.tools.insta_tools import (
    InstaTool,
    InstaGradioTool
)

from insta.tools.transformers_tools import (
    InstaTransformersTool,
    InstaTransformersGradioTool
)

from insta.tools.langchain_tools import (
    InstaLangchainTool,
    InstaLangchainGradioTool
)


TOOLS = {
    'insta': InstaTool,
    'insta-gradio': InstaGradioTool,
    'insta-transformers': InstaTransformersTool,
    'insta-transformers-gradio': InstaTransformersGradioTool,
    'insta-langchain': InstaLangchainTool,
    'insta-langchain-gradio': InstaLangchainGradioTool
}
