#!/bin/bash

source /miniconda3/bin/activate
conda activate insta

touch playwright.log && tail -f playwright.log &
bash start_playwright_server.sh

python gradio/app.py