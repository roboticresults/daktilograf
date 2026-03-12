#!/bin/bash

# Audio Input Device Checker
# This script checks for available audio input devices (microphones)

echo "=========================================="
echo "Audio Input Device Checker"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if virtual environment exists
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Error: Virtual environment not found at $SCRIPT_DIR/.venv"
    echo "Please run: uv venv"
    exit 1
fi

# Activate virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Run Python script to check audio devices
python3 << 'EOF'
import sounddevice as sd
import sys

print("Checking for audio input devices...")
print("=" * 50)

try:
    # Get all audio devices
    devices = sd.query_devices()
    
    # Filter for input devices (microphones)
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device))
    
    if not input_devices:
        print("❌ No audio input devices found!")
        print("\nTroubleshooting:")
        print("1. Check if a microphone is connected")
        print("2. Check system audio settings")
        print("3. On Linux, check if you have the right permissions")
        print("4. Try: sudo usermod -a -G audio $USER")
        sys.exit(1)
    
    print(f"✅ Found {len(input_devices)} audio input device(s):\n")
    
    for idx, device in input_devices:
        print(f"Device {idx}: {device['name']}")
        print(f"  Channels: {device['max_input_channels']}")
        print(f"  Sample Rate: {device['default_samplerate']} Hz")
        print(f"  Latency: low={device['default_low_input_latency']:.3f}s, high={device['default_high_input_latency']:.3f}s")
        print()
    
    # Set default device
    default_input = sd.default.device[0]
    if default_input is not None:
        print(f"Default input device: {devices[default_input]['name']}")
    else:
        print(f"Default input device: {devices[input_devices[0][0]]['name']}")
    
    print("\n" + "=" * 50)
    print("✅ Audio input devices are ready!")
    
except Exception as e:
    print(f"❌ Error checking audio devices: {e}")
    print("\nTroubleshooting:")
    print("1. Install PortAudio: sudo apt-get install portaudio19-dev (Linux)")
    print("2. Check microphone connections")
    print("3. Verify system audio settings")
    sys.exit(1)
EOF

EXIT_CODE=$?

deactivate 2>/dev/null || true

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Audio device check completed successfully"
else
    echo "❌ Audio device check failed"
fi
echo "=========================================="

# Wait for user to press Enter before closing
echo ""
echo "Press Enter to close this window..."
read -r

exit $EXIT_CODE
