#!/usr/bin/env python3
"""
Hotkey listener module for managing recording state via keyboard shortcuts.
"""

import threading
from pynput.keyboard import Listener, Key, KeyCode


class HotkeyListener:
    """
    Manages hotkey listener for starting/stopping recording.
    Uses threading.Event for thread-safe state management.
    
    Implements a simpler approach by tracking key states directly
    instead of using pynput's HotKey class which has issues on Linux.
    """
    
    # The hotkey combination: Ctrl+Alt+Shift+G
    HOTKEY_KEYS = {Key.ctrl, Key.alt, Key.shift}
    HOTKEY_CHAR = 'g'
    
    def __init__(self, on_toggle_callback=None):
        """
        Initialize hotkey listener.
        
        Args:
            on_toggle_callback: Optional callback function when hotkey is pressed.
                               The callback receives the new recording state (bool).
        """
        self._recording_event = threading.Event()
        self._listener = None
        self._on_toggle_callback = on_toggle_callback
        
        # Track currently pressed keys
        self._current_keys = set()
        self._hotkey_triggered = False
        
        # Lock for thread-safe key tracking
        self._key_lock = threading.Lock()
    
    def _on_activate_hotkey(self):
        """Callback function when hotkey is pressed."""
        # Toggle the recording state
        new_state = not self._recording_event.is_set()
        self._recording_event.set() if new_state else self._recording_event.clear()
        
        # Print visual feedback
        if new_state:
            print("\n🎤 Recording started with hotkey!")
        else:
            print("\n🛑 Recording stopped with hotkey!")
        
        # Call external callback if provided
        if self._on_toggle_callback:
            self._on_toggle_callback(new_state)
    
    def _on_press(self, key):
        """Handle key press events."""
        with self._key_lock:
            # Normalize the key
            key = self._normalize_key(key)
            
            # Add to current keys
            self._current_keys.add(key)
            
            # Check if all modifier keys are pressed
            if self._all_modifiers_pressed():
                # Check if 'g' is also pressed and we haven't triggered yet
                if self._has_g_key() and not self._hotkey_triggered:
                    self._hotkey_triggered = True
                    self._on_activate_hotkey()
    
    def _on_release(self, key):
        """Handle key release events."""
        with self._key_lock:
            # Normalize the key
            key = self._normalize_key(key)
            
            # Remove from current keys
            self._current_keys.discard(key)
            
            # Reset trigger if any modifier is released
            if key in self.HOTKEY_KEYS or self._is_g_key(key):
                self._hotkey_triggered = False
    
    def _normalize_key(self, key):
        """Normalize key to handle both Key and KeyCode."""
        if isinstance(key, KeyCode):
            # Convert to lowercase for comparison
            if key.char:
                return KeyCode.from_char(key.char.lower())
        return key
    
    def _all_modifiers_pressed(self):
        """Check if all modifier keys (Ctrl, Alt, Shift) are pressed."""
        # Check for left or right variants of modifier keys
        has_ctrl = Key.ctrl in self._current_keys or Key.ctrl_l in self._current_keys or Key.ctrl_r in self._current_keys
        has_alt = Key.alt in self._current_keys or Key.alt_l in self._current_keys or Key.alt_r in self._current_keys
        has_shift = Key.shift in self._current_keys or Key.shift_l in self._current_keys or Key.shift_r in self._current_keys
        
        return has_ctrl and has_alt and has_shift
    
    def _has_g_key(self):
        """Check if 'g' key is pressed."""
        g_key = KeyCode.from_char('g')
        return g_key in self._current_keys
    
    def _is_g_key(self, key):
        """Check if a key is the 'g' key."""
        if isinstance(key, KeyCode) and key.char:
            return key.char.lower() == 'g'
        return False
    
    def start(self):
        """Start the hotkey listener in a background thread."""
        print("🎧 Setting up hotkey listener (Ctrl+Alt+Shift+G)...")
        print("   Press Ctrl+Alt+Shift+G to start/stop recording")
        
        # Create the listener
        self._listener = Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        
        # Start the listener as a daemon thread
        self._listener.start()
        return self
    
    def stop(self):
        """Stop the hotkey listener."""
        if self._listener:
            self._listener.stop()
            self._listener = None
    
    @property
    def recording_active(self):
        """Check if recording is currently active."""
        return self._recording_event.is_set()
    
    @recording_active.setter
    def recording_active(self, value):
        """Set the recording state."""
        if value:
            self._recording_event.set()
        else:
            self._recording_event.clear()
