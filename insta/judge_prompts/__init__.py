from insta.judge_prompts.base_judge_prompt import (
    BaseJudgePrompt
)

from insta.judge_prompts.verbose_judge_prompt import (
    VerboseJudgePrompt
)


JUDGE_PROMPTS = {
    'base': BaseJudgePrompt,
    'verbose': VerboseJudgePrompt,
}