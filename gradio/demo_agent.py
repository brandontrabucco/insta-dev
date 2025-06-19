from insta import (
    get_agent_config,
    get_judge_config,
    get_browser_config,
    InstaPipeline,
    InstaPipelineOutput
)

from typing import Tuple
from datetime import datetime

import gradio
import argparse

import numpy as np
import cv2
import skvideo.io

import os
import re

DEFAULT_VIDEO_HEIGHT = 720
DEFAULT_VIDEO_WIDTH = 1280

DEFAULT_MAX_ACTIONS = 30
DEFAULT_NUM_SAMPLES = 8

DEFAULT_MAX_OBS_TOKENS = 2048
DEFAULT_LAST_OBS = 3

DEFAULT_SUCCESS = 0.0


ID_PATTERN = re.compile(
    r"(backend_node_id)\s*=\s*([\"\'])(\d+)\2"
)


def create_video(
    url: str, instruction: str,
    trajectory: InstaPipelineOutput,
    output_height: int = DEFAULT_VIDEO_HEIGHT,
    output_width: int = DEFAULT_VIDEO_WIDTH,
) -> np.ndarray:

    frames = []

    for obs, act in zip(
        trajectory.observations,
        trajectory.actions,
    ):
        
        has_screenshot = (
            "screenshot" in obs and
            obs["screenshot"] is not None
        )
        
        if not has_screenshot:
            
            continue

        metadata = obs["metadata"]

        frame = np.asarray(obs["screenshot"])
        frame_height, frame_width = frame.shape[:2]

        # Resize the frame

        output_shape = (
            output_width,
            output_height
        )

        frame = cv2.resize(
            frame, output_shape,
            interpolation = cv2.INTER_AREA
        )

        downsize_factor_x = (
            frame_width / output_width
        )

        downsize_factor_y = (
            frame_height / output_height
        )

        frame_height, frame_width = frame.shape[:2]

        # add a white bar to the top of the frame and add the task in green font
        # expand the frame by 30 pixels down

        bottom_space = int(30 / 360 * frame_height)
        task_spacing = int(25 / 360 * frame_height)
        action_spacing = int(45 / 360 * frame_height)

        frame = np.concatenate([
            frame, np.ones((
                bottom_space * 2, frame_width, 3
            ), np.uint8) * 255,
        ], axis = 0)

        frame = cv2.putText(
            frame,
            "Instruction: {}".format(instruction),
            (10, frame_height + task_spacing),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 255, 0),
            2
        )

        frames.extend(
            [frame] * 24 * 1
        )

        frame = frame.copy()

        clean_action = (
            act["matched_response"]
            .replace("    ", "")
            .replace("\n", " ")
            .replace("\r", " ")
            .replace("\t", " ")
        )

        frame = cv2.putText(
            frame,
            "Action: {}".format(clean_action),
            (10, frame_height + action_spacing),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (255, 0, 0),
            2
        )
        
        match = ID_PATTERN.search(
            act["function_calls"][0]["args"]
        )

        bounding_client_rect = None

        if match is not None:

            node_metadata = metadata.get(
                match.group(3)
            )

            if node_metadata is not None:

                bounding_client_rect = node_metadata.get(
                    "bounding_client_rect"
                )

        if bounding_client_rect is not None:

            x = bounding_client_rect["x"]
            y = bounding_client_rect["y"]
            width = bounding_client_rect["width"]
            height = bounding_client_rect["height"]

            x = int(x / downsize_factor_x)
            y = int(y / downsize_factor_y)
            width = int(width / downsize_factor_x)
            height = int(height / downsize_factor_y)

            frame = cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (255, 0, 0),
                2
            )

        frames.extend(
            [frame] * 24 * 3
        )

    return np.stack(frames, axis = 0)


NULL_VIDEO = None


def generate_trajectory(
    url: str, instruction: str,
    output_height: int = DEFAULT_VIDEO_HEIGHT,
    output_width: int = DEFAULT_VIDEO_WIDTH,
) -> Tuple[str, str]:

    if instruction is None or len(instruction) == 0:

        return NULL_VIDEO, "No task was entered"

    url = (url or "https://duckduckgo.com/")

    has_protocol = (
        url.startswith("http://")
        or url.startswith("https://")
    )

    if not has_protocol:

        url = "https://{}".format(url)

    print("Collecting {n} Trajectories\nInitial URL: {url}\nInstruction: {instruction}".format(
        n = args.num_samples,
        url = url, instruction = instruction
    ))

    instruction_dataset = [
        {"domain": url, "task": instruction}
    ] * args.num_samples

    trajectories = pipeline.launch(
        dataset = instruction_dataset,
        num_agents = args.num_agents,
        playwright_workers = args.playwright_workers,
        return_trajectories = True
    )

    print("Finished Data Collection")

    def get_trajectory_success(x):

        if x is None or not isinstance(x, InstaPipelineOutput):

            return DEFAULT_SUCCESS

        trajectory_valid = (
            x.observations is not None and 
            len(x.observations) > 0 and 
            x.actions is not None and 
            len(x.actions) > 0 and 
            x.judgment is not None and 
            len(x.judgment) > 0
        )

        if not trajectory_valid:
            
            return DEFAULT_SUCCESS

        current_success = (
            x.judgment.get("success") or
            DEFAULT_SUCCESS
        )

        return current_success

    best_trajectory = max(
        trajectories,
        key = get_trajectory_success
    )

    agent_succeeded = (
        best_trajectory.judgment is not None 
        and (best_trajectory.judgment.get("success") or DEFAULT_SUCCESS) > 0.5
    )

    if agent_succeeded:

        action_summary = "\n\n---\n\n".join([
            x["response"]
            for x in best_trajectory.actions
        ])

        video_frames = create_video(
            url = url, instruction = instruction,
            trajectory = best_trajectory,
            output_height = output_height,
            output_width = output_width
        )

        print("Num Frames = {}".format(
            len(video_frames)
        ))

        os.makedirs(
            "demo/videos",
            exist_ok = True
        )

        timestamp = (
            datetime.now()
            .strftime('%Y-%m-%d %H:%M:%S')
        )

        target_video_path = (
            "demo/videos/{}-{:06d}.mp4"
            .format(timestamp, np.random.randint(999999))
        )

        skvideo.io.vwrite(
            target_video_path,
            video_frames
        )

        return target_video_path, action_summary
    
    return NULL_VIDEO, "Agent did not succeed on this task"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent_model_name",
        type = str,
        default = "btrabucco/Insta-Qwen2.5-1.5B-SFT"
    )

    parser.add_argument(
        "--agent_client_type",
        type = str,
        default = "openai"
    )

    parser.add_argument(
        "--agent_llm_endpoint",
        type = str,
        default = "http://localhost:8000/v1"
    )

    parser.add_argument(
        "--agent_api_key",
        type = str,
        default = "token-abc123",
    )

    parser.add_argument(
        "--judge_model_name",
        type = str,
        default = "gpt-4.1-nano"
    )

    parser.add_argument(
        "--judge_client_type",
        type = str,
        default = "openai"
    )

    parser.add_argument(
        "--judge_llm_endpoint",
        type = str,
        default = "https://api.openai.com/v1"
    )

    parser.add_argument(
        "--judge_api_key",
        type = str,
        default = os.environ.get("OPENAI_API_KEY")
    )

    parser.add_argument(
        "--agent_prompt",
        type = str,
        default = "simplified_json"
    )

    parser.add_argument(
        "--max_obs_tokens",
        type = int,
        help = "Maximum number of tokens per observations",
        default = DEFAULT_MAX_OBS_TOKENS
    )

    parser.add_argument(
        "--last_obs",
        type = int,
        help = "Previous observations in context",
        default = DEFAULT_LAST_OBS
    )

    parser.add_argument(
        "--max_actions",
        type = int,
        help = "Maximum number of actions to take",
        default = DEFAULT_MAX_ACTIONS
    )

    parser.add_argument(
        "--agent_response_key",
        type = str,
        help = "key for response from the agent",
        default = "response",
    )

    parser.add_argument(
        "--playwright_url",
        type = str,
        help = "URL template for Playwright server",
        default = "http://localhost:{port}"
    )

    parser.add_argument(
        "--playwright_port",
        type = int,
        help = "Port for Playwright server",
        default = 3000
    )

    parser.add_argument(
        "--playwright_workers",
        type = int,
        help = "Number of parallel Playwright servers",
        default = 8
    )

    parser.add_argument(
        "--num_agents",
        type = int,
        help = "Number of parallel agents to run",
        default = DEFAULT_NUM_SAMPLES
    )

    parser.add_argument(
        "--num_samples",
        type = int,
        help = "Number of samples to select from",
        default = DEFAULT_NUM_SAMPLES
    )

    args = parser.parse_args()

    args.num_agents = min(
        args.num_agents,
        args.num_samples
    )

    agent_client_type = args.agent_client_type

    agent_client_kwargs = {}

    agent_generation_kwargs = {
        "max_tokens": 2048,
        "top_p": 1.0,
        "temperature": 0.5
    }

    if args.agent_client_type == "openai":

        agent_generation_kwargs.update({
            "model": args.agent_model_name
        })

    if args.agent_client_type == "vllm":

        agent_client_kwargs.update({
            "model": args.agent_model_name
        })

    if args.agent_api_key is not None:

        agent_client_kwargs.update({
            "api_key": args.agent_api_key
        })

    if args.agent_llm_endpoint is not None:

        agent_client_kwargs.update({
            "base_url": args.agent_llm_endpoint
        })

    agent_config = get_agent_config(
        client_type = agent_client_type,
        client_kwargs = agent_client_kwargs,
        generation_kwargs = agent_generation_kwargs,
        agent_prompt = args.agent_prompt,
        max_obs_tokens = args.max_obs_tokens,
        last_obs = args.last_obs,
    )
    
    judge_client_type = args.judge_client_type

    judge_client_kwargs = {}

    judge_generation_kwargs = {
        "max_tokens": 2048,
        "top_p": 1.0,
        "temperature": 0.5
    }

    if args.judge_client_type == "openai":

        judge_generation_kwargs.update({
            "model": args.judge_model_name
        })

    if args.judge_client_type == "vllm":

        judge_client_kwargs.update({
            "model": args.judge_model_name
        })

    if args.judge_api_key is not None:

        judge_client_kwargs.update({
            "api_key": args.judge_api_key
        })

    if args.judge_llm_endpoint is not None:

        judge_client_kwargs.update({
            "base_url": args.judge_llm_endpoint
        })

    judge_config = get_judge_config(
        client_type = judge_client_type,
        client_kwargs = judge_client_kwargs,
        generation_kwargs = judge_generation_kwargs,
    )

    browser_config = get_browser_config(
        playwright_url = args.playwright_url,
        playwright_port = args.playwright_port
    )

    pipeline = InstaPipeline(
        agent_config = agent_config,
        judge_config = judge_config,
        browser_config = browser_config,
        agent_response_key = args.agent_response_key,
        max_actions = args.max_actions,
        observations_dir = None,
        screenshot_dir = None,
        actions_dir = None,
        judgments_dir = None
    )

    url_textbox = gradio.Textbox(
        label = "Initial URL",
        placeholder = "Enter an initial URL ...",
        value = "https://duckduckgo.com/"
    )

    instruction_textbox = gradio.Textbox(    
        label = "Instruction",
        placeholder = "Enter an instruction ..."
    )

    output_video = gradio.Video(
        label = "Agent Video"
    )

    output_textbox = gradio.Textbox(
        label = "Agent Response",
        placeholder = "Final agent response ..."
    )

    gradio_app = gradio.Interface(
        generate_trajectory,
        inputs = [
            url_textbox,
            instruction_textbox,
        ],
        outputs = [
            output_video,
            output_textbox,
        ],
        title = "LLM Agent App",
    )

    gradio_app.launch()
