#!/bin/bash

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
"${THIS_DIR}"/docker.sh --image_suffix="forever" --forever $@