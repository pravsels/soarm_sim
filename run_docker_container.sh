#!/bin/bash
IMAGE_NAME="soarm_sim"

xhost +local:root

docker run --gpus all --rm -it --network=host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video,display \
  -e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
  -v /etc/vulkan/icd.d:/etc/vulkan/icd.d:ro \
  -v "$(pwd)":/workspace \
  "$IMAGE_NAME"
