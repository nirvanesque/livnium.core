#!/bin/bash
# Run integration examples with proper PYTHONPATH

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the repo root (parent of scripts directory)
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Set PYTHONPATH to include the repo root
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# Run the example
python3 "$REPO_ROOT/livnium/examples/document_pipeline_example.py"

