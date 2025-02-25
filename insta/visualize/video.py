from multiprocessing import Pool
from functools import partial
from datasets import load_dataset

from typing import Tuple, List, Dict

import argparse
import tqdm
import json

import os
import re

import skvideo.io
import cv2
import numpy as np


ID_PATTERN = re.compile(
    r"(backend_node_id)\s*=\s*([\"\'])(\d+)\2"
)


def load_trajectory(
    target_file_name: str,
    observations_dir: str,
    actions_dir: str,
    judgments_dir: str
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    
    observations_path = os.path.join(
        observations_dir,
        target_file_name
    )

    actions_path = os.path.join(
        actions_dir,
        target_file_name
    )

    judgments_path = os.path.join(
        judgments_dir,
        target_file_name
    )

    with open(observations_path, "r") as file:
        observations = json.load(file)

    with open(actions_path, "r") as file:
        actions = json.load(file)

    with open(judgments_path, "r") as file:
        judgment = json.load(file)

    return observations, actions, judgment


def create_video(
    target_file_name: str,
    domain_to_task: dict,
    observations_dir: str = "data/observations",
    actions_dir: str = "data/actions",
    judgments_dir: str = "data/judgments",
    video_dir: str = "data/videos",
    output_height: int = 720,
    output_width: int = 1280,
    task_is_feasible_threshold: float = 1.0,
    success_threshold: float = 1.0,
    on_right_track_threshold: float = 1.0,
) -> str:

    domain = target_file_name.replace(".json", "")
    task = domain_to_task[domain]

    observations, actions, judgment = load_trajectory(
        target_file_name,
        observations_dir,
        actions_dir,
        judgments_dir
    )

    trajectory_in_threshold = (
        judgment["task_is_feasible"] is not None and
        judgment["task_is_feasible"]  >= task_is_feasible_threshold and
        judgment["success"] is not None and
        judgment["success"] >= success_threshold and
        judgment["on_right_track"] is not None and
        judgment["on_right_track"] >= on_right_track_threshold
    )

    if not trajectory_in_threshold:

        return domain

    frames = []

    for obs, act in zip(
        observations,
        actions,
    ):
        
        screenshot_path = obs["screenshot_path"]
        metadata = obs["metadata"]

        if screenshot_path is None or \
                not os.path.exists(screenshot_path):
            
            continue

        frame = cv2.imread(
            screenshot_path
        )

        frame_height, frame_width, _ = frame.shape

        # Resize the frame

        frame = cv2.resize(
            frame,
            (output_width, output_height),
            interpolation = cv2.INTER_AREA
        )

        downsize_factor_x = frame_width / output_width
        downsize_factor_y = frame_height / output_height

        frame_height, frame_width, _ = frame.shape

        matched_response = ""

        for function_call in act["function_calls"]:

            if len(matched_response) > 0:

                matched_response += "."

            matched_response += "{}({})".format(
                function_call["dotpath"],
                function_call["args"]
            )

        # add a white bar to the top of the frame and add the task in green font
        # expand the frame by 30 pixels down

        bottom_space = int(30 / 360 * frame_height)
        task_spacing = int(25 / 360 * frame_height)
        action_spacing = int(45 / 360 * frame_height)

        frame = np.concatenate([
            frame,
            np.ones((bottom_space * 2, frame_width, 3), np.uint8) * 255,
        ], axis = 0)

        frame = cv2.putText(
            frame,
            "Task: {}".format(task),
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

        frame = cv2.putText(
            frame,
            "Action: {}".format(matched_response),
            (10, frame_height + action_spacing),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (255, 0, 0),
            2
        )
        
        id_match = ID_PATTERN.search(
            matched_response
        )

        if id_match is not None:
            
            id_value = id_match.group(3)

            if id_value is not None:

                node_metadata = metadata.get(id_value, None)

                if node_metadata is not None:

                    bounding_client_rect = node_metadata.get(
                        "bounding_client_rect", None
                    )

                    if bounding_client_rect is not None:

                        x = bounding_client_rect["x"]
                        y = bounding_client_rect["y"]
                        width = bounding_client_rect["width"]
                        height = bounding_client_rect["height"]

                        x = x / downsize_factor_x
                        y = y / downsize_factor_y
                        width = width / downsize_factor_x
                        height = height / downsize_factor_y

                        x = int(x)
                        y = int(y)
                        width = int(width)
                        height = int(height)

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

    if len(frames) == 0:

        return domain

    writer = skvideo.io.FFmpegWriter(
        os.path.join(
            video_dir,
            domain + ".mp4"
        )
    )

    for frame in frames:
        writer.writeFrame(frame)

    writer.close()

    return domain


def create_demo_videos(
    dataset: str = "data-for-agents/insta-150k",
    dataset_split: str = "train",
    observations_dir: str = "data/observations",
    actions_dir: str = "data/actions",
    judgments_dir: str = "data/judgments",
    video_dir: str = "data/videos",
    output_height: int = 720,
    output_width: int = 1280,
    task_is_feasible_threshold: float = 1.0,
    success_threshold: float = 1.0,
    on_right_track_threshold: float = 1.0,
):

    dataset = load_dataset(
        dataset, split = dataset_split
    )

    domain_to_task = {
        x["domain"]: x["task"]
        for x in dataset
    }

    os.makedirs(
        video_dir,
        exist_ok = True
    )

    all_files = os.listdir(actions_dir)

    progress_bar = tqdm.tqdm(
        all_files,
        desc = "Creating videos",
        dynamic_ncols = True
    )

    worker_fn = partial(
        create_video,
        domain_to_task = domain_to_task,
        observations_dir = observations_dir,
        actions_dir = actions_dir,
        judgments_dir = judgments_dir,
        video_dir = video_dir,
        output_height = output_height,
        output_width = output_width,
        task_is_feasible_threshold = task_is_feasible_threshold,
        success_threshold = success_threshold,
        on_right_track_threshold = on_right_track_threshold
    )

    with Pool(processes = 32) as pool:
        
        for domain in pool.imap_unordered(
            worker_fn, all_files
        ):

            progress_bar.set_description(
                "Processing: {}".format(domain)
            )
            
            progress_bar.update()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type = str,
        default = "data-for-agents/insta-150k",
    )
    
    parser.add_argument(
        "--observations_dir",
        type = str,
        default = "data/observations"
    )
    
    parser.add_argument(
        "--actions_dir",
        type = str,
        default = "data/actions"
    )
    
    parser.add_argument(
        "--judgments_dir",
        type = str,
        default = "data/judgments"
    )

    parser.add_argument(
        "--video_dir",
        type = str,
        default = "data/videos"
    )
    
    parser.add_argument(
        "--output_height",
        type = int,
        default = 720
    )

    parser.add_argument(
        "--output_width",
        type = int,
        default = 1280
    )

    parser.add_argument(
        "--task_is_feasible_threshold",
        type = float,
        default = 1.0
    )

    parser.add_argument(
        "--success_threshold",
        type = float,
        default = 1.0
    )

    parser.add_argument(
        "--on_right_track_threshold",
        type = float,
        default = 1.0
    )

    args = parser.parse_args()
    
    create_demo_videos(
        dataset = args.dataset,
        observations_dir = args.observations_dir,
        actions_dir = args.actions_dir,
        judgments_dir = args.judgments_dir,
        video_dir = args.video_dir,
        output_height = args.output_height,
        output_width = args.output_width,
        task_is_feasible_threshold = args.task_is_feasible_threshold,
        success_threshold = args.success_threshold,
        on_right_track_threshold = args.on_right_track_threshold
    )
