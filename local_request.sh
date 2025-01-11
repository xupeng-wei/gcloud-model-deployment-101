#!/bin/bash

INPUT_DATA_FILE='/home/jupyter/demo-pipeline-storage/demo_project/app/request_demo.json'
PORT=9090

curl \
-X POST "http://localhost:${PORT}/predict" \
-H "Content-Type: application/json" \
-d "@${INPUT_DATA_FILE}"