from insta.judgment_parsers.judgement_parser import (
    BaseJudgmentParser
)

from insta.judgment_parsers.json_judgment_parser import (
    JsonJudgmentParser
)

from insta.judgment_parsers.simplified_json_judgment_parser import (
    SimplifiedJsonJudgmentParser
)


JUDGMENT_PARSERS = {
    'json': JsonJudgmentParser,
    'simplified_json': SimplifiedJsonJudgmentParser,
}