#!/bin/bash
# Run integration examples with proper PYTHONPATH

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set PYTHONPATH to include the repo root
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Run the example
python3 "$SCRIPT_DIR/livnium/examples/document_pipeline_example.py"

