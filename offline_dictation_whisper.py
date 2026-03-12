#!/usr/bin/env python3
"""
Offline Speech-to-Text Dictation Tool using Whisper.cpp

This script provides real-time speech recognition using Whisper.cpp
and automatically types the recognized text into the active window.

Usage:
    python offline_dictation_whisper.py [--model MODEL_PATH] [--log]

Arguments:
    --model MODEL_PATH  Path to Whisper ggml model (default: ./models/ggml-base.bin)
    --log               Enable logging to offline_dictate_log.txt
    --typing-interval   Time between keystrokes in seconds (default: 0.07)
    --pause             Pause between phrases in seconds (default: 0.5)

Requirements:
    - pywhispercpp
    - sounddevice
    - pyautogui
    - numpy
    - pynput (for hotkey detection)

Setup:
    1. Download a Whisper ggml model (e.g., ggml-base.bin)
    2. Place it in the ./models/ directory or specify path with --model
    3. Install requirements: pip install -r requirements.txt
    4. Run: python offline_dictation_whisper.py

Safety:
    - Move mouse to top-left corner to abort (pyautogui failsafe)
    - Press Ctrl+C to stop the program gracefully

# New Feature: Real-time Segment Processing
#
# This version introduces support for real-time processing of transcription
# segments via the `new_segment_callback` parameter.
#
# How it works:
# - The `transcribe_audio` function now accepts an optional
#   `new_segment_callback` argument.
# - When provided, each recognized segment is passed to the callback,
#   allowing immediate typing and logging.
# - The callback receives a `Segment` object with `.text`, `.start`, and
#   `.end` attributes.
# - Typical callback actions include:
#   * Typing the segment text via `pyautogui` with natural timing.
#   * Logging the text to a file for record-keeping.
# - If no callback is provided, the original behavior (transcribe the
#   entire recording at once) is preserved.
#
# Example callback:
#     def _real_time_callback(segment):
#         if segment.text and len(segment.text) > 2:
#             type_text(segment.text, typing_interval)
#             log_transcription(segment.text)
#
# The callback is automatically invoked by the transcription loop when
# `new_segment_callback` is supplied.
"""

import argparse
import sys
import time
import datetime
import os
import threading
from pathlib import Path

import sounddevice as sd
import pyautogui
import numpy as np
from pywhispercpp.model import Model

# Import from modules
from modules.config import Config
from modules.audio_processor import AudioProcessor
from modules.hotkey_listener import HotkeyListener
from modules.realtime_dictation import RealtimeDictation


# ============================================================================
# AUDIO DEVICE UTILITIES
# ============================================================================

def list_audio_devices():
    """List available audio input devices with detailed information."""
    try:
        devices = sd.query_devices()
        print("\nAvailable audio input devices:")
        print("=" * 70)
        input_devices = []
        has_usb_headset_output_only = False
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append(i)
                print(f"\nDevice {i}: {device['name']}")
                print(f"  Channels: {device['max_input_channels']}")
                print(f"  Sample Rate: {device['default_samplerate']} Hz")
                print(f"  Latency: low={device['default_low_input_latency']:.3f}s, high={device['default_high_input_latency']:.3f}s")
                
                # Check if this is the default input device
                default_input = sd.default.device[0]
                if default_input is not None and i == default_input:
                    print("  *** This is the DEFAULT input device ***")
                
                # Identify device type based on name patterns
                device_name_lower = device['name'].lower()
                if 'hw:' in device['name'] or 'hardware' in device_name_lower:
                    print("  Type: Hardware device (direct access)")
                elif 'pulse' in device_name_lower:
                    print("  Type: PulseAudio (system audio)")
                elif 'default' in device_name_lower:
                    print("  Type: Default system device")
                elif 'sysdefault' in device_name_lower:
                    print("  Type: System default device")
                elif 'usb' in device_name_lower:
                    print("  Type: USB audio device")
                elif 'hdmi' in device_name_lower:
                    print("  Type: HDMI audio")
                elif 'hda intel' in device_name_lower or 'alc' in device_name_lower:
                    print("  Type: Built-in audio codec (3.5mm jack)")
        
        # Check for USB headsets that are output-only (common issue)
        for i, device in enumerate(devices):
            device_name_lower = device['name'].lower()
            if ('headset' in device_name_lower or 'usb' in device_name_lower) and device['max_input_channels'] == 0 and device['max_output_channels'] > 0:
                has_usb_headset_output_only = True
                print(f"\n⚠️  Note: Device {i} '{device['name']}' is output-only (no microphone input)")
                print("   This USB headset's microphone may be routed through the system default device.")
                print("   To use this headset, select it as the default input in your system audio settings,")
                print("   then use the 'default' or 'sysdefault' device in this application.")
        
        print("\n" + "=" * 70)
        print(f"Total: {len(input_devices)} input device(s) found")
        print("=" * 70)
        
        if has_usb_headset_output_only:
            print("\n💡 TIP: If using a USB headset, make sure it's selected as the default input device")
            print("   in your system audio settings, then use the 'default' device in this application.")
        
        return input_devices
    except Exception as error:
        print(f"Error listing audio devices: {error}")
        return []

def select_audio_device():
    """Select audio device - uses system default device for USB headset compatibility."""
    print("\nDetecting audio devices...")
    
    # List available devices
    devices = list_audio_devices()
    
    if not devices:
        print("No input devices found!")
        return None
    
    all_devices = sd.query_devices()
    
    # Always use the system default input device
    # This ensures USB headsets work correctly when selected in system audio settings
    # Note: sd.default.device is a tuple (input_device, output_device)
    default_input = sd.default.device[0]
    
    if default_input is not None and default_input in devices:
        print(f"\n✓ Using system default input device: {all_devices[default_input]['name']}")
        print("  (This respects your system audio settings - make sure your USB headset is selected there)")
        return default_input
    
    # Fallback: Use the first available input device
    print(f"\n⚠️  No default input device set, using first available: {all_devices[devices[0]]['name']}")
    return devices[0]

def load_whisper_model(model_path):
    """
    Load Whisper.cpp model from file.
    
    Args:
        model_path: Path to ggml model file
        
    Returns:
        Loaded Whisper model
    """
    try:
        print(f"Loading Whisper model from {model_path}...")
        model = Model(str(model_path), n_threads=8)        
        print("✓ Model loaded successfully")
        return model
    except Exception as error:
        print(f"✗ Failed to load model: {error}")
        print("\nPlease download a Whisper ggml model from:")
        print("  https://huggingface.co/ggerganov/whisper.cpp")
        sys.exit(1)


# ============================================================================
# SPEECH RECOGNITION
# ============================================================================

def transcribe_audio(model, audio_data, sample_rate, new_segment_callback=None, typing_interval=0.07, enable_logging=False):
    """
    Transcribe audio using Whisper model.
    
    Args:
        model: Loaded Whisper model
        audio_data: Normalized audio data
        sample_rate: Audio sample rate
        new_segment_callback: Optional callback called for each new segment
        typing_interval: Time between keystrokes
        enable_logging: Whether to log transcriptions
        
    Returns:
        Transcribed text or None if no speech detected
    """
    try:
        # Ensure audio is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Prepare kwargs for transcribe
        kwargs = {}
        if new_segment_callback is not None:
            kwargs['new_segment_callback'] = new_segment_callback
        
        # Transcribe with Serbian language
        result = model.transcribe(audio_data, **kwargs, n_processors=1)
        
        # Process results - ONLY use callback OR return text, not both
        if result and len(result) > 0:
            if new_segment_callback is not None:
                # Callback already handled typing during transcription
                # Just return the combined text without typing again
                texts = [seg.text.strip() for seg in result if seg.text and len(seg.text.strip()) > 2]
                text = " ".join(texts) if texts else ""
            else:
                # No callback - type the text now
                text = result[0].text.strip()
                if text and len(text) > 2:
                    type_text(text, typing_interval)
                    if enable_logging:
                        log_transcription(text)
            
            if text and len(text) > 2:
                return text
        
        return None
        
    except Exception as error:
        print(f"✗ Transcription failed: {error}")
        return None


# ============================================================================
# TEXT OUTPUT
# ============================================================================

def type_text(text, typing_interval):
    """
    Type text using pyautogui with natural typing speed.
    
    Args:
        text: Text to type
        typing_interval: Seconds between keystrokes
    """
    try:
        print(f"✓ Typing: {text}")
        pyautogui.typewrite(text, interval=typing_interval)
        pyautogui.typewrite(" ")  # Add trailing space
    except pyautogui.FailSafeException:
        raise
    except Exception as error:
        print(f"✗ Typing failed: {error}")


def log_transcription(text):
    """Log transcribed text to file with timestamp."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(Config.LOG_FILE, 'a', encoding='utf-8') as log_file:
            log_file.write(f"[{timestamp}] {text}\n")
    except Exception as error:
        print(f"✗ Logging failed: {error}")


# ============================================================================
# MAIN RECOGNITION LOOP
# ============================================================================

def run_dictation_loop(model, audio_processor, hotkey_listener, typing_interval, pause_duration, enable_logging):
    """
    Main dictation loop - continuously records audio while hotkey is active,
    then processes the entire recording when hotkey is released.
    
    Args:
        model: Loaded Whisper model
        audio_processor: AudioProcessor instance
        hotkey_listener: HotkeyListener instance
        typing_interval: Seconds between keystrokes
        pause_duration: Seconds to pause between phrases
        enable_logging: Whether to log transcriptions
    """
    last_recording_state = False

    is_processing = False
    audio_stream = None
    
    print("\n" + "="*60)
    print("Dictation started! Speak clearly into your microphone.")
    print("Move mouse to top-left corner to abort (failsafe).")
    print("Press Ctrl+C to stop.")
    print("="*60 + "\n")
    
    try:
        while True:
            # Check if recording state has changed
            current_recording_state = hotkey_listener.recording_active
            
            if current_recording_state != last_recording_state:
                if current_recording_state:
                    # Recording started - start capturing audio
                    print("\n🔴 RECORDING - Speak now...")
                    audio_processor.start_continuous_recording()
                    
                    # Start audio stream for continuous recording
                    device_sample_rate = audio_processor._get_device_sample_rate()
                    block_size = int(device_sample_rate * 0.1)  # 100ms blocks
                    
                    audio_stream = sd.InputStream(
                        samplerate=device_sample_rate,
                        channels=1,
                        dtype='int16',
                        device=audio_processor.device_id,
                        callback=audio_processor._audio_callback
                    )
                    audio_stream.start()
                else:
                     # Recording stopped - process the recorded audio
                    if is_processing:
                         # Already processing a previous stop, just update state and continue
                        last_recording_state = current_recording_state
                        continue
                    is_processing = True
                    try:
                        print("\n⚫ STOPPED - Processing audio...")
                    
                        # Stop the audio stream
                        if audio_stream:
                            audio_stream.stop()
                            audio_stream.close()
                            audio_stream = None
                        
                        # Stop recording and get the audio data
                        audio_processor.stop_continuous_recording()
                        audio_data, duration = audio_processor.get_recorded_audio()
                        
                        if len(audio_data) > 0:
                            print(f"  Recorded {duration:.1f} seconds of audio")
                            
                            # Check if audio is long enough
                            if duration >= Config.MIN_AUDIO_DURATION:
                                # Transcribe the entire recording
                                # Define real-time segment callback - types text as it comes
                                def _real_time_segment_callback(segment):
                                    if segment.text and len(segment.text) > 2:
                                        type_text(segment.text, typing_interval)
                                        if enable_logging:
                                            log_transcription(segment.text)
                                
                                # Transcribe - callback handles typing in real-time
                                transcribe_audio(
                                    model,
                                    audio_data,
                                    audio_processor.target_sample_rate,
                                    new_segment_callback=_real_time_segment_callback,
                                    typing_interval=typing_interval,
                                    enable_logging=enable_logging
                                )
                                
                                # Pause between phrases
                                time.sleep(pause_duration)
                            else:
                                print(f"  Audio too short ({duration:.1f}s < {Config.MIN_AUDIO_DURATION}s)")
                        else:
                            print("  No speech detected in recording")
                    except KeyboardInterrupt:
                        if audio_stream:
                            audio_stream.stop()
                            audio_stream.close()
                            audio_stream = None
                        print("\n\nDiktiranje prekinuto.")
                    # Clear the buffer for next recording
                    audio_processor.clear_buffer()
                    print("⚫ STOPPED - Waiting for hotkey (Ctrl+Alt+Shift+G)...")
                
                last_recording_state = current_recording_state
            
            # Small delay to prevent CPU overuse
            time.sleep(0.05)
            
   
    except pyautogui.FailSafeException:
        print("\n\nFailsafe triggered! Program aborted.")
    finally:
        # Clean up audio stream if still active
        if audio_stream:
            audio_stream.stop()
            audio_stream.close()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Offline Speech-to-Text Dictation Tool using Whisper.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live toggle mode (default): Press Ctrl+Alt+Shift+G to start/stop live dictation
  python offline_dictation_whisper.py
  
  # With Serbian language model
  python offline_dictation_whisper.py --language sr
  
  # With logging
  python offline_dictation_whisper.py --model ./models/ggml-small.bin --log
  
  # Adjust VAD sensitivity (0=least strict, 3=most strict)
  python offline_dictation_whisper.py --vad-aggressiveness 2
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="./models/ggml-base.bin",
        help="Path to Whisper ggml model"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=['hotkey', 'realtime'],
        default='hotkey',
        help="Dictation mode: 'hotkey' (toggle live dictation with hotkey) or 'realtime' (start immediately with VAD)"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default=Config.DEFAULT_LANGUAGE,
        help="Language code (e.g., 'sr' for Serbian, 'en' for English)"
    )
    
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable logging to offline_dictate_log.txt"
    )
    
    parser.add_argument(
        "--typing-interval",
        type=float,
        default=Config.DEFAULT_TYPING_INTERVAL,
        help="Time between keystrokes in seconds"
    )
    
    parser.add_argument(
        "--pause",
        type=float,
        default=Config.DEFAULT_PAUSE_BETWEEN_PHRASES,
        help="Pause between phrases in seconds (hotkey mode only)"
    )
    
    # VAD options (used in both modes now)
    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        default=Config.VAD_AGGRESSIVENESS,
        choices=[0, 1, 2, 3],
        help="VAD aggressiveness: 0=least strict, 3=most strict"
    )
    
    # Transcription quality options
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for beam search (higher = better quality, slower). Default: 5"
    )
    
    parser.add_argument(
        "--best-of",
        type=int,
        default=5,
        help="Number of candidates for sampling (higher = better quality). Default: 5"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy/best, higher = more random). Default: 0.0"
    )
    
    return parser.parse_args()


def run_realtime_mode(args):
    """
    Run dictation in real-time VAD mode.
    
    Uses Voice Activity Detection for automatic speech transcription.
    """
    print("\n" + "="*60)
    print("Real-time Dictation Mode (VAD-based)")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Language: {args.language}")
    print(f"VAD Aggressiveness: {args.vad_aggressiveness}")
    print("="*60)
    
    # Load Whisper model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        print("\nPlease download a Whisper ggml model:")
        print("  - ggml-base.bin: https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin")
        print("  - ggml-small.bin: https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin")
        sys.exit(1)
    
    # Text callback for typing
    def on_text_recognized(text):
        """Called when text is recognized."""
        type_text(text, args.typing_interval)
        if args.log:
            log_transcription(text)
    
    # Create and start real-time dictation
    try:
        dictation = RealtimeDictation(
            model_path=str(model_path),
            text_callback=on_text_recognized,
            language=args.language,
            n_threads=8,
            vad_aggressiveness=args.vad_aggressiveness,
            print_progress=True
        )
        dictation.start()
    except KeyboardInterrupt:
        print("\n\nStopping...")
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        print("\nProgram terminated.")


def get_optimal_threads():
    """Get optimal number of threads for Whisper inference."""
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 4  # Default fallback
    # Use all available threads for maximum performance
    return cpu_count


def check_gpu_available():
    """Check if GPU acceleration is available."""
    try:
        # Try to import and check for CUDA
        import ctypes
        # Check if CUDA is available via ctypes
        try:
            cuda = ctypes.CDLL('libcudart.so')
            return True, "CUDA"
        except OSError:
            pass
        
        # Check for ROCm (AMD GPU)
        try:
            rocm = ctypes.CDLL('libamdhip64.so')
            return True, "ROCm"
        except OSError:
            pass
        
        # Check for OpenCL
        try:
            cl = ctypes.CDLL('libOpenCL.so')
            return True, "OpenCL"
        except OSError:
            pass
            
    except Exception:
        pass
    
    return False, None


def run_hotkey_mode(args):
    """
    Run dictation in toggle-based live mode.
    
    Press hotkey once to start live transcription, press again to stop.
    Uses Voice Activity Detection for automatic speech detection.
    """
    print("\n" + "="*60)
    print("Live Dictation Mode (Toggle)")
    print("="*60)
    
    # Get optimal thread count
    n_threads = get_optimal_threads()
    gpu_available, gpu_type = check_gpu_available()
    
    print(f"Model: {args.model}")
    print(f"Language: {args.language}")
    print(f"Threads: {n_threads} (max)")
    print(f"GPU: {gpu_type if gpu_available else 'Not available (using CPU)'}")
    print(f"Beam size: {args.beam_size} | Best of: {args.best_of} | Temperature: {args.temperature}")
    print("="*60)
    
    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        print("\nPlease download a Whisper ggml model:")
        print("  - ggml-base.bin: https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin")
        print("  - ggml-small.bin: https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin")
        sys.exit(1)
    
    # Select audio device
    device_id = select_audio_device()
    
    # State for dictation
    dictation = None
    dictation_thread = None
    is_running = False
    stop_event = threading.Event()
    
    # Text callback for typing
    def on_text_recognized(text):
        """Called when text is recognized."""
        type_text(text, args.typing_interval)
        if args.log:
            log_transcription(text)
    
    def run_dictation():
        """Run dictation in a separate thread."""
        nonlocal dictation
        try:
            dictation = RealtimeDictation(
                model_path=str(model_path),
                text_callback=on_text_recognized,
                language=args.language,
                input_device=device_id,
                n_threads=n_threads,
                vad_aggressiveness=args.vad_aggressiveness,
                print_progress=False,
                beam_size=args.beam_size,
                best_of=args.best_of,
                temperature=args.temperature
            )
            dictation.start()
        except Exception as e:
            if not stop_event.is_set():
                print(f"\n✗ Dictation error: {e}")
    
    def on_hotkey_toggle(active):
        """Handle hotkey toggle."""
        nonlocal dictation_thread, is_running
        
        if active and not is_running:
            # Start dictation
            print("\n🎤 LIVE MODE ACTIVATED - Start speaking...")
            print("   (Press Ctrl+Alt+Shift+G to stop)")
            is_running = True
            stop_event.clear()
            
            # Start dictation in a separate thread
            dictation_thread = threading.Thread(target=run_dictation)
            dictation_thread.daemon = True
            dictation_thread.start()
            
        elif not active and is_running:
            # Stop dictation
            print("\n🛑 STOPPING live dictation...")
            is_running = False
            stop_event.set()
            
            if dictation:
                dictation.stop()
            
            if dictation_thread and dictation_thread.is_alive():
                dictation_thread.join(timeout=2.0)
            
            print("   Live dictation stopped. Press Ctrl+Alt+Shift+G to start again.")
    
    # Start hotkey listener with toggle callback
    hotkey_listener = HotkeyListener(on_toggle_callback=on_hotkey_toggle)
    hotkey_listener.start()
    
    # Prompt user
    print("\n" + "="*60)
    print("Ready! Press Ctrl+Alt+Shift+G to START live dictation.")
    print("Press Ctrl+Alt+Shift+G again to STOP.")
    print("Press Ctrl+C to exit the program.")
    print("="*60 + "\n")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        # Cleanup
        if is_running and dictation:
            dictation.stop()
        hotkey_listener.stop()
        print("Program terminated.")


def main():
    """Main application entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0
    
    # Print header
    print("="*60)
    print("Offline Speech-to-Text Dictation Tool")
    print("Using Whisper.cpp for offline speech recognition")
    print("="*60)
    
    # Run in selected mode
    if args.mode == 'realtime':
        run_realtime_mode(args)
    else:
        run_hotkey_mode(args)


if __name__ == "__main__":
    main()
