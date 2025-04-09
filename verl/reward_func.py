from insta import (
    ACTION_PARSERS,
    BaseActionParser,
    BrowserStatus,
)

from insta.utils import safe_call

import json


SIMPLIFIED_JSON_PARSER: BaseActionParser = (
    ACTION_PARSERS["simplified_json"]()
)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info = None
) -> float:
    """Reward function for training LLM agents to operate a browser,
    and complete a desired web navigation task.

    Arguments:

    model_output: str
        The model output to evaluate.

    ground_truth: str
        The ground truth output to compare against.

    Returns:

    reward: float
        The reward value for GRPO update.
    
    """
    
    ground_truth = json.loads(
        ground_truth
    )

    action = safe_call(
        SIMPLIFIED_JSON_PARSER.parse_action,
        response = solution_str,
        catch_errors = True,
        max_errors = 1,
        log_errors = False
    )

    if action is BrowserStatus.ERROR:

        return 0.0

    action = json.loads(
        action.matched_response
    )

    has_required_keys = (
        "action_key" in action and
        "target_element_id" in action and
        "action_kwargs" in action
    )

    if not has_required_keys:

        return 0.0

    reward = 0.1

    action_key_match = (
        ground_truth["action_key"] == 
        action["action_key"]
    )

    if action_key_match:

        reward += 0.3

    target_element_match = (
        ground_truth["target_element_id"] == 
        action["target_element_id"]
    )

    if target_element_match:

        reward += 0.3

    ground_truth_kwargs_str = json.dumps(
        ground_truth["action_kwargs"]
    )

    action_kwargs_str = json.dumps(
        action["action_kwargs"]
    )

    action_kwargs_match = (
        ground_truth_kwargs_str ==
        action_kwargs_str
    )

    if action_kwargs_match:

        reward += 0.3

    return reward