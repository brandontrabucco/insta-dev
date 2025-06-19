from insta.task_proposer_prompts.base_task_proposer_prompt import (
    BaseTaskProposerPrompt
)

from insta.task_proposer_prompts.verbose_task_proposer_prompt import (
    VerboseTaskProposerPrompt
)

from insta.task_proposer_prompts.task_explorer_prompt import (
    TaskExplorerPrompt
)

from insta.task_proposer_prompts.task_refiner_prompt import (
    TaskRefinerPrompt
)

from insta.task_proposer_prompts.task_composer_prompt import (
    TaskComposerPrompt
)


TASK_PROPOSER_PROMPTS = {
    'base': BaseTaskProposerPrompt,
    'verbose': VerboseTaskProposerPrompt,
    'explorer': TaskExplorerPrompt,
    'refiner': TaskRefinerPrompt,
    'composer': TaskComposerPrompt,
}