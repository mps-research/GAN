#!/usr/bin/env bash

docker run \
    -v $(pwd)/src:/code \
    -v $(pwd)/data:/data \
    -v $(pwd)/logs:/logs \
    -v $(pwd)/models:/models \
    -p 6006:6006 \
    --gpus all \
    --shm-size 3G \
    -it gan python3 run.py --train