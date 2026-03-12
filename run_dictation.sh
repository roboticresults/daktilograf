#!/bin/bash

# Offline Dictation Tool Launcher Script
# This script activates the virtual environment and runs the dictation tool
# with the small model and logging enabled.

set -e  # Exit on error

echo "=========================================="
echo "Offline Speech-to-Text Dictation Tool"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Working directory: $SCRIPT_DIR"
echo ""

# Check if virtual environment exists
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Error: Virtual environment not found at $SCRIPT_DIR/.venv"
    echo "Please run: uv venv"
    exit 1
fi

# Check if model exists
MODEL_PATH="$SCRIPT_DIR/model/ggml-small.bin"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please download the model and place it in the model/ directory"
    exit 1
fi

echo "✓ Virtual environment found"
echo "✓ Model found: $MODEL_PATH"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source "$SCRIPT_DIR/.venv/bin/activate"
echo "✓ Virtual environment activated"
echo ""

# Run the dictation tool
echo "Starting dictation tool with live dictation mode..."
echo ""
echo "Instructions:"
echo "  1. Press Ctrl+Alt+Shift+G to START live dictation"
echo "  2. Speak - text will be typed automatically"
echo "  3. Press Ctrl+Alt+Shift+G again to STOP"
echo "  4. Press Ctrl+C to exit the program"
echo ""
echo "Safety: Move mouse to top-left corner to abort (failsafe)"
echo "=========================================="
echo ""

# Run the Python script with specified options
python "$SCRIPT_DIR/offline_dictation_whisper.py" \
    --model "$MODEL_PATH" \
    --log \
    "$@"

# Capture the exit code
EXIT_CODE=$?

# Deactivate virtual environment (though this happens automatically when script exits)
deactivate 2>/dev/null || true

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Dictation tool finished successfully"
else
    echo "Dictation tool exited with code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
