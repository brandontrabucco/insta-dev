from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class JudgeConfig:

    tokenizer: str = "meta-llama/Llama-3.3-70B-Instruct"
    client_kwargs: Dict = None
    generation_kwargs: Dict = None

    last_actions: int = 5
    last_obs: int = 1
    max_obs_tokens: int = 4096

    catch_errors: bool = True
    max_errors: int = 5
    log_errors: bool = True


DEFAULT_TOKENIZER = "meta-llama/Llama-3.3-70B-Instruct"


DEFAULT_CLIENT_KWARGS = {
    "api_key": "token-abc123",
    "base_url": "http://localhost:8000/v1",
}


DEFAULT_GENERATION_KWARGS = {
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "max_tokens": 2048,
    "top_p": 1.0,
    "temperature": 0.0
}


DEFAULT_LAST_ACTIONS = 5
DEFAULT_LAST_OBS = 1
DEFAULT_MAX_OBS_TOKENS = 4096


DEFAULT_CATCH_ERRORS = True
DEFAULT_MAX_ERRORS = 5
DEFAULT_LOG_ERRORS = False


DEFAULT_JUDGE_CONFIG = JudgeConfig(
    tokenizer = DEFAULT_TOKENIZER,
    client_kwargs = DEFAULT_CLIENT_KWARGS,
    generation_kwargs = DEFAULT_GENERATION_KWARGS,
    last_actions = DEFAULT_LAST_ACTIONS,
    last_obs = DEFAULT_LAST_OBS,
    max_obs_tokens = DEFAULT_MAX_OBS_TOKENS,
    catch_errors = DEFAULT_CATCH_ERRORS,
    max_errors = DEFAULT_MAX_ERRORS,
    log_errors = DEFAULT_LOG_ERRORS,
)


def get_judge_config(
    **judge_kwargs
) -> JudgeConfig:
    
    default_judge_kwargs = asdict(DEFAULT_JUDGE_CONFIG)
    default_judge_kwargs.update(judge_kwargs)
    
    return JudgeConfig(
        **default_judge_kwargs
    )
