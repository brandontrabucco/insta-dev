from insta.task_parsers.base_task_parser import (
    BaseTaskParser
)
from insta.task_parsers.json_task_parser import (
    JsonTaskParser
)


TASK_PARSERS = {
    'json': JsonTaskParser,
}