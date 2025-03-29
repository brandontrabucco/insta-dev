from insta.judgment_parsers.judgement_parser import (
    BaseJudgmentParser
)

from insta.judgment_parsers.json_judgment_parser import (
    JsonJudgmentParser
)


JUDGMENT_PARSERS = {
    'json': JsonJudgmentParser,
}