#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
python -m qv2x demo --config configs/demo.json --output runs/demo
