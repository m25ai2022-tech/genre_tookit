#!/bin/bash
# Start the demo server; open the browser once it is actually up (startup takes ~25 s).
cd "$(dirname "$0")"
( until curl -s http://127.0.0.1:5057/api/info >/dev/null 2>&1; do sleep 1; done; open "http://127.0.0.1:5057" ) &
python3 app.py
