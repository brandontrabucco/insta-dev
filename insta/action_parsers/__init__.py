from insta.action_parsers.base_action_parser import (
    BaseActionParser
)
from insta.action_parsers.json_action_parser import (
    JsonActionParser
)
from insta.action_parsers.javascript_action_parser import (
    JavascriptActionParser
)


ACTION_PARSERS = {
    'json': JsonActionParser,
    'javascript': JavascriptActionParser
}