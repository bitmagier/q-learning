#!/bin/bash

docker run -u "$(id -u):$(id -g)" --gpus all -it --rm --mount type=bind,source="$(pwd)",target=/rl-breakout tensorflow/tensorflow:2.12.0-gpu bash
