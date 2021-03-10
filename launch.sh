#!/bin/bash

docker run \
  --rm \
  -v /usr/local/fsl:/usr/local/fsl \
  -v /usr/local/freesurfer:/usr/local/freesurfer \
  -v /data:/data \
  --user "$(id -u $USER):$(id -g $USER)" \
  -v $(pwd):/app \
  -w /app \
  mrigroupopbg/mri-python3:libraries python3 $@

