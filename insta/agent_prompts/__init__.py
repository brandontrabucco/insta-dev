from insta.agent_prompts.base_agent_prompt import (
    BaseAgentPrompt
)

from insta.agent_prompts.verbose_agent_prompt import (
    VerboseAgentPrompt
)


AGENT_PROMPTS = {
    'base': BaseAgentPrompt,
    'verbose': VerboseAgentPrompt
}