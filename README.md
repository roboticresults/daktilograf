# Offline Dictation Tool

A Python-based offline speech recognition system using Whisper.cpp for multi-language speech processing. Enables real-time transcription without internet connectivity.

## Model Setup
1. Download Whisper ggml model:
   - Base model (147 MB): [ggml-base.bin](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin)
   - Small model (466 MB): [ggml-small.bin](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin)
2. Place the model file in the `model/` directory

## Installation
```bash
uv venv
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\activate  # Windows
uv pip install -r requirements.txt
```

## Quick Start (Recommended)
Use the provided launcher scripts for easy execution:

```bash
# Using small model (better accuracy)
./run_dictation.sh

# Using base model (lighter and faster)
./run_dictation_base.sh
```

## Manual Usage
```bash
# Using base model (default)
python offline_dictation_whisper.py

# Using small model for better accuracy
python offline_dictation_whisper.py --model ./model/ggml-small.bin

# With logging enabled
python offline_dictation_whisper.py --log

# Custom typing speed and pause
python offline_dictation_whisper.py --typing-interval 0.05 --pause 0.3
```

## Features
- **Failsafe**: Move mouse to top-left corner to immediately terminate
- **Logging**: All transcriptions saved to `offline_dictate_log.txt` with timestamps
- Real-time speech recognition in multiple languages (including Serbian)
- Cross-platform compatibility (Windows, macOS, Linux)
- Configurable typing speed and pause between phrases
- Automatic phrase detection and typing

## Launcher Scripts
For convenience, two executable launcher scripts are provided:

- **`run_dictation.sh`** - Uses the small model (ggml-small.bin) for better accuracy
- **`run_dictation_base.sh`** - Uses the base model (ggml-base.bin) for faster performance

Both scripts:
- Automatically activate the virtual environment
- Enable logging to `offline_dictate_log.txt`
- Handle cleanup and error reporting
- Can be run from any directory

## Requirements
- Python 3.8+
- Microphone input device
- At least 200MB free RAM for base model, 500MB for small model
- pynput library (automatically installed with requirements.txt)

## Safety
- **IMPORTANT**: The script enables `pyautogui.FAILSAFE = True`
- Move mouse to the top-left corner of screen to abort immediately
- Press Ctrl+C for graceful shutdown

## Features
- **Hotkey Control**: Press Ctrl+Alt+Shift+G to start/stop recording
- **Automatic Microphone Detection**: Automatically detects and selects the best available microphone
- **Hotkey Feedback**: Visual indicators when recording starts/stops

## Supported Languages
Whisper.cpp supports 99 languages including Serbian, English, Spanish, French, German, and many more. The model automatically detects the spoken language.