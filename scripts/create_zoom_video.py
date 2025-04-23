from PIL import Image

import skvideo.io
import cv2
import numpy as np

import argparse
import glob
import os

import tqdm


TARGET_LENGTH = 24 * 15
SPEED = 4
NUM_ROWS = 13
BUFFER_FRACTION = 0.1


if __name__ == "__main__":

    video_files = glob.glob(
        "data-v3/videos/*.mp4"
    )

    video_tensors = []

    progress_bar = tqdm.tqdm(
        video_files[:(NUM_ROWS * NUM_ROWS)],
        desc = "Loading videos"
    )

    for video_file in progress_bar:

        video_data = skvideo.io.vread(
            video_file
        )

        video_tensors.append(
            video_data
        )

    output_video = []

    for frame in range(TARGET_LENGTH):

        all_frames = [
            video_data[(frame * SPEED) % video_data.shape[0]]
            for video_data in video_tensors
        ]

        frame_rows = []

        for row in range(NUM_ROWS):

            frame_cols = []

            for col in range(NUM_ROWS):

                frame_cols.append(
                    all_frames.pop()
                )

            frame_cols = np.concatenate(
                frame_cols,
                axis = 1
            )

            frame_rows.append(
                frame_cols
            )

        frame_rows = np.concatenate(
            frame_rows,
            axis = 0
        )

        percentage = (
            (1 / NUM_ROWS) + (1 - (1 / NUM_ROWS)) * 
            max(0, frame - TARGET_LENGTH * BUFFER_FRACTION) / (TARGET_LENGTH * (1 - BUFFER_FRACTION))
        )

        target_width = frame_rows.shape[1] * percentage
        target_height = frame_rows.shape[0] * percentage

        offset_left = frame_rows.shape[1] * (1 - percentage) / 2
        offset_top = frame_rows.shape[0] * (1 - percentage) / 2
        
        target_width = int(target_width)
        target_height = int(target_height)

        offset_left = int(offset_left)
        offset_top = int(offset_top)

        sliced_frame = frame_rows[
            offset_top:offset_top + target_height,
            offset_left:offset_left + target_width
        ]

        sliced_frame = Image.fromarray(
            sliced_frame
        )

        sliced_frame = sliced_frame.resize(
            (640, 360)
        )

        sliced_frame = np.asarray(
            sliced_frame
        )

        output_video.append(
            sliced_frame
        )

    output_video = np.stack(
        output_video,
        axis = 0
    )

    skvideo.io.vwrite(
        "web_agent_zoom.mp4", 
        output_video
    )
