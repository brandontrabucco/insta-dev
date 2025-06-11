from insta.task_parsers.task_parser import (
    BaseTaskParser
)

from insta.task_parsers.json_task_parser import (
    JsonTaskParser
)

from insta.task_parsers.simplified_json_task_parser import (
    SimplifiedJsonTaskParser
)

from insta.task_parsers.refiner_json_task_parser import (
    RefinerJsonTaskParser
)


TASK_PARSERS = {
    'json': JsonTaskParser,
    'simplified_json': SimplifiedJsonTaskParser,
    'refiner_json': RefinerJsonTaskParser,
}