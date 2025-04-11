from dataclasses import dataclass, asdict
from typing import Dict, List
from insta.configs.browser_config import FunctionCall


@dataclass
class AgentConfig:

    action_parser: str = "json"

    tokenizer: str = "meta-llama/Llama-3.3-70B-Instruct"

    client_type: str = None
    client_kwargs: Dict = None
    generation_kwargs: Dict = None

    last_obs: int = 5
    max_obs_tokens: int = 4096

    catch_errors: bool = True
    log_errors: bool = True
    max_errors: int = 5


@dataclass
class BrowserAction:

    function_calls: List[FunctionCall] = None
    response: str = None
    matched_response: str = None


DEFAULT_ACTION_PARSER = "json"


DEFAULT_TOKENIZER = "meta-llama/Llama-3.3-70B-Instruct"


DEFAULT_CLIENT_TYPE = "openai"


DEFAULT_CLIENT_KWARGS = {
    "api_key": "token-abc123",
    "base_url": "http://localhost:8000/v1",
}


DEFAULT_GENERATION_KWARGS = {
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "max_tokens": 2048,
    "top_p": 1.0,
    "temperature": 0.5
}


DEFAULT_LAST_OBS = 5
DEFAULT_MAX_OBS_TOKENS = 4096


DEFAULT_CATCH_ERRORS = True
DEFAULT_MAX_ERRORS = 5
DEFAULT_LOG_ERRORS = False


DEFAULT_AGENT_CONFIG = AgentConfig(
    action_parser = DEFAULT_ACTION_PARSER,
    tokenizer = DEFAULT_TOKENIZER,
    client_type = DEFAULT_CLIENT_TYPE,
    client_kwargs = DEFAULT_CLIENT_KWARGS,
    generation_kwargs = DEFAULT_GENERATION_KWARGS,
    last_obs = DEFAULT_LAST_OBS,
    max_obs_tokens = DEFAULT_MAX_OBS_TOKENS,
    catch_errors = DEFAULT_CATCH_ERRORS,
    max_errors = DEFAULT_MAX_ERRORS,
    log_errors = DEFAULT_LOG_ERRORS,
)


def get_agent_config(
    **agent_kwargs
) -> AgentConfig:
    
    default_agent_kwargs = asdict(DEFAULT_AGENT_CONFIG)
    default_agent_kwargs.update(agent_kwargs)
    
    return AgentConfig(
        **default_agent_kwargs
    )
