#!/bin/bash
docker run --gpus all -it --rm --mount type=bind,source="$(pwd)",target=/rl-breakout tensorflow/tensorflow:latest-gpu bash