from insta.judgment_parsers.base_judgement_parser import (
    BaseJudgmentParser
)
from insta.judgment_parsers.json_judgment_parser import (
    JsonJudgmentParser
)
from insta.judgment_parsers.javascript_judgment_parser import (
    JavascriptJudgmentParser
)


JUDGMENT_PARSERS = {
    'json': JsonJudgmentParser,
    'javascript': JavascriptJudgmentParser
}