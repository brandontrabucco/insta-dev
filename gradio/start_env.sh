#!/bin/bash

export SERVER_SCRIPT=${SERVER_SCRIPT:-"javascript/server/src/index.js"}
export SERVER_LOG=${SERVER_LOG:-"gradio/playwright.log"}

export SERVER_BASE_PORT=${SERVER_BASE_PORT:-3000}
export SERVER_WORKERS=${SERVER_WORKERS:-8}
export MAX_ERRORS=${MAX_ERRORS:-1000}

source /miniconda3/bin/activate
conda activate insta

rm ${SERVER_LOG}
touch ${SERVER_LOG}

bash start_playwright_server.sh
tail -f ${SERVER_LOG} &

python -u gradio/demo_env.py