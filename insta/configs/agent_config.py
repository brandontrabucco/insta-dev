from dataclasses import dataclass
from typing import Dict


@dataclass
class AgentConfig:

    tokenizer: str = "meta-llama/Llama-3.3-70B-Instruct"
    client_kwargs: Dict = None
    generation_kwargs: Dict = None

    max_history: int = 5
    max_obs_tokens: int = 4096

    catch_errors: bool = True
    log_errors: bool = True
    max_errors: int = 5


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


DEFAULT_MAX_HISTORY = 5
DEFAULT_MAX_OBS_TOKENS = 4096


DEFAULT_CATCH_ERRORS = True
DEFAULT_MAX_ERRORS = 5
DEFAULT_LOG_ERRORS = False


DEFAULT_AGENT_CONFIG = AgentConfig(
    tokenizer = DEFAULT_TOKENIZER,
    client_kwargs = DEFAULT_CLIENT_KWARGS,
    generation_kwargs = DEFAULT_GENERATION_KWARGS,
    max_history = DEFAULT_MAX_HISTORY,
    max_obs_tokens = DEFAULT_MAX_OBS_TOKENS,
    catch_errors = DEFAULT_CATCH_ERRORS,
    max_errors = DEFAULT_MAX_ERRORS,
    log_errors = DEFAULT_LOG_ERRORS,
)


def get_agent_config(
    **agent_config_kwargs
) -> AgentConfig:
    
    return AgentConfig(
        tokenizer = agent_config_kwargs.get(
            "tokenizer",
            DEFAULT_TOKENIZER
        ),
        client_kwargs = agent_config_kwargs.get(
            "client_kwargs",
            DEFAULT_CLIENT_KWARGS
        ),
        generation_kwargs = agent_config_kwargs.get(
            "generation_kwargs",
            DEFAULT_GENERATION_KWARGS
        ),
        max_history = agent_config_kwargs.get(
            "max_history",
            DEFAULT_MAX_HISTORY
        ),
        max_obs_tokens = agent_config_kwargs.get(
            "max_obs_tokens",
            DEFAULT_MAX_OBS_TOKENS
        ),
        catch_errors = agent_config_kwargs.get(
            "catch_errors", 
            DEFAULT_CATCH_ERRORS
        ),
        max_errors = agent_config_kwargs.get(
            "max_errors", 
            DEFAULT_MAX_ERRORS
        ),
        log_errors = agent_config_kwargs.get(
            "log_errors", 
            DEFAULT_LOG_ERRORS
        ),
    )

