from insta.action_parsers.action_parser import (
    BaseActionParser
)

from insta.action_parsers.json_action_parser import (
    JsonActionParser
)

from insta.action_parsers.simplified_json_action_parser import (
    SimplifiedJsonActionParser
)


ACTION_PARSERS = {
    'json': JsonActionParser,
    'simplified_json': SimplifiedJsonActionParser
}