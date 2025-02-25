from multiprocessing import Pool
from functools import partial
from datasets import load_dataset

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


def create_video(
    target_file_name: str,
    action_parser = None,
    args = None,
    domain_to_task: dict = None
):

    domain = target_file_name.replace(".json", "")
    task = domain_to_task[domain]

    actions_path = os.path.join(
        args.actions_dir,
        target_file_name
    )

    observations_path = os.path.join(
        args.observations_dir,
        target_file_name
    )

    with open(actions_path, "r") as f:
        actions = json.load(f)

    with open(observations_path, "r") as f:
        observations = json.load(f)

    frames = []

    for obs, act in zip(
        observations,
        actions
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
            (args.output_width, args.output_height),
            interpolation = cv2.INTER_AREA
        )

        downsize_factor_x = frame_width / args.output_width
        downsize_factor_y = frame_height / args.output_height

        frame_height, frame_width, _ = frame.shape

        matched_response = act["matched_response"]
        matched_args = " ".join(x["args"] for x in act["function_calls"])

        # add a white bar to the top of the frame and add the task in green font
        # expand the frame by 30 pixels down

        bottom_space = int(30 / 360 * frame_height)
        task_spacing = int(20 / 360 * frame_height)
        action_spacing = int(50 / 360 * frame_height)

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
            matched_args
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
            args.video_dir,
            domain + ".mp4"
        )
    )

    for frame in frames:
        writer.writeFrame(frame)

    writer.close()

    return domain


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
        "--screenshot_dir",
        type = str,
        default = "data/screenshots"
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

    args = parser.parse_args()

    dataset = load_dataset(
        args.dataset, split = "train"
    )

    domain_to_task = {
        x["domain"]: x["task"]
        for x in dataset
    }

    os.makedirs(
        args.video_dir,
        exist_ok = True
    )

    all_files = os.listdir(args.actions_dir)

    progress_bar = tqdm.tqdm(
        all_files,
        desc = "Creating videos",
        dynamic_ncols = True
    )

    worker_fn = partial(
        create_video,
        args = args,
        domain_to_task = domain_to_task
    )

    with Pool(processes = 32) as pool:
        
        for domain in pool.imap_unordered(
            worker_fn, all_files
        ):

            progress_bar.set_description(
                "Processing: {}".format(domain)
            )
            
            progress_bar.update()
