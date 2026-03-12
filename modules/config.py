#!/usr/bin/env python3
"""
Configuration constants for the Offline Dictation Tool.
"""

from pynput.keyboard import Key, KeyCode


class Config:
    """Application configuration constants."""
    
    # Audio settings
    SAMPLE_RATE = 16000  # Whisper expects 16kHz
    BLOCK_DURATION = 15.0  # Seconds of audio per capture block
    
    # Audio processing
    AMPLIFICATION_FACTOR = 20.0  # Boost quiet microphone signal
    
    # Typing settings
    DEFAULT_TYPING_INTERVAL = 0.07  # Seconds between keystrokes
    DEFAULT_PAUSE_BETWEEN_PHRASES = 0.5  # Seconds between phrases
    
    # Recognition settings
    MIN_AUDIO_DURATION = 1.0  # Minimum seconds of audio to process
    
    # Logging
    LOG_FILE = "offline_dictate_log.txt"
    
    # Hotkey settings - Ctrl+Alt+Shift+G
    HOTKEY_COMBINATION = [
        Key.ctrl,
        Key.alt,
        Key.shift,
        KeyCode.from_char('g')
    ]
    
    # Real-time dictation settings (VAD-based)
    VAD_SILENCE_THRESHOLD = 8  # Frames of silence before transcribing
    VAD_QUEUE_THRESHOLD = 16  # Minimum audio chunks before transcribing
    VAD_BLOCK_DURATION_MS = 30  # Audio block duration in milliseconds
    VAD_AGGRESSIVENESS = 1  # VAD aggressiveness (0-3, higher = more strict)
    
    # Language settings
    DEFAULT_LANGUAGE = 'sr'  # Serbian (use 'en' for English)
