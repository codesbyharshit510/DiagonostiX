#!/usr/bin/env bash
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/matplotlib
mkdir -p /tmp/matplotlib
uvicorn main:app --host 0.0.0.0 --port 10000
