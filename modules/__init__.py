"""
Modules package for Offline Dictation Tool.

This package provides modular components for speech-to-text dictation:
- Config: Configuration constants
- AudioProcessor: Audio capture from microphone
- HotkeyListener: Hotkey management for recording control
- RealtimeDictation: Voice Activity Detection based real-time transcription
"""

from modules.config import Config
from modules.audio_processor import AudioProcessor
from modules.hotkey_listener import HotkeyListener

# RealtimeDictation is imported separately as it requires webrtcvad
try:
    from modules.realtime_dictation import RealtimeDictation
    __all__ = ['Config', 'AudioProcessor', 'HotkeyListener', 'RealtimeDictation']
except ImportError:
    __all__ = ['Config', 'AudioProcessor', 'HotkeyListener']
