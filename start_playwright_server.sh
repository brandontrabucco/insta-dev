#!/bin/bash

export SERVER_DIR=${SERVER_DIR:-"javascript/server"}
export SERVER_SCRIPT=${SERVER_SCRIPT:-"javascript/server/src/index.js"}
export SERVER_LOG=${SERVER_LOG:-"playwright.log"}

export SERVER_BASE_PORT=${SERVER_BASE_PORT:-3000}
export SERVER_WORKERS=${SERVER_WORKERS:-8}
export MAX_ERRORS=${MAX_ERRORS:-1000}

read -r -d '' PLAYWRIGHT_COMMAND << END_OF_SCRIPT

pushd ${SERVER_DIR} && npx tsc && popd

seq 1 ${MAX_ERRORS} | xargs --process-slot-var WORKER_IDX -I {} -P ${SERVER_WORKERS} \
    bash -c "node ${SERVER_SCRIPT} \\\$((WORKER_IDX + ${SERVER_BASE_PORT})) >> ${SERVER_LOG} 2>&1"

END_OF_SCRIPT

# start a new playwright server in the background
screen -S playwright -dm \
    bash -c "${PLAYWRIGHT_COMMAND}"
