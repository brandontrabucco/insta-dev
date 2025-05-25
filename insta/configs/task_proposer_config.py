from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class TaskProposerConfig:

    task_parser: str = "json"

    tokenizer: str = "Qwen/Qwen2.5-72B-Instruct"

    client_type: str = None
    client_kwargs: Dict = None
    generation_kwargs: Dict = None

    last_judgments: int = 5
    last_tasks: int = 5
    last_trajectories: int = 1

    last_actions: int = 5
    last_obs: int = 5
    max_obs_tokens: int = 2048

    catch_errors: bool = True
    max_errors: int = 5
    log_errors: bool = True


@dataclass
class BrowserTaskProposal:

    proposed_task: str = None
    steps: List[str] = None
    criteria: List[str] = None

    response: str = None
    matched_response: str = None


DEFAULT_TASK_PARSER = "json"


DEFAULT_TOKENIZER = "Qwen/Qwen2.5-72B-Instruct"


DEFAULT_CLIENT_TYPE = "openai"


DEFAULT_CLIENT_KWARGS = {
    "api_key": "token-abc123",
    "base_url": "http://localhost:8000/v1",
}


DEFAULT_GENERATION_KWARGS = {
    "model": "Qwen/Qwen2.5-72B-Instruct",
    "max_tokens": 1024,
    "top_p": 1.0,
    "temperature": 0.5
}


DEFAULT_LAST_JUDGMENTS = 5
DEFAULT_LAST_TASKS = 5
DEFAULT_LAST_TRAJECTORIES = 1


DEFAULT_LAST_ACTIONS = 5
DEFAULT_LAST_OBS = 5
DEFAULT_MAX_OBS_TOKENS = 2048


DEFAULT_CATCH_ERRORS = True
DEFAULT_MAX_ERRORS = 5
DEFAULT_LOG_ERRORS = False


DEFAULT_TASK_PROPOSER_CONFIG = TaskProposerConfig(
    task_parser = DEFAULT_TASK_PARSER,
    tokenizer = DEFAULT_TOKENIZER,
    client_type = DEFAULT_CLIENT_TYPE,
    client_kwargs = DEFAULT_CLIENT_KWARGS,
    generation_kwargs = DEFAULT_GENERATION_KWARGS,
    last_judgments = DEFAULT_LAST_JUDGMENTS,
    last_tasks = DEFAULT_LAST_TASKS,
    last_trajectories = DEFAULT_LAST_TRAJECTORIES,
    last_actions = DEFAULT_LAST_ACTIONS,
    last_obs = DEFAULT_LAST_OBS,
    max_obs_tokens = DEFAULT_MAX_OBS_TOKENS,
    catch_errors = DEFAULT_CATCH_ERRORS,
    max_errors = DEFAULT_MAX_ERRORS,
    log_errors = DEFAULT_LOG_ERRORS,
)


def get_task_proposer_config(
    **task_proposer_kwargs
) -> TaskProposerConfig:
    
    default_task_proposer_kwargs = asdict(DEFAULT_TASK_PROPOSER_CONFIG)
    default_task_proposer_kwargs.update(task_proposer_kwargs)

    return TaskProposerConfig(
        **default_task_proposer_kwargs
    )
