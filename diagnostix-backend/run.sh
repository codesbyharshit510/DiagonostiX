#!/usr/bin/env bash
# Simple helper to run the dev server
export PYTHONPATH=.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
