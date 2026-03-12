#!/usr/bin/env python3
"""
Audio processor module for capturing microphone input.
Handles continuous recording with sample rate conversion for Whisper compatibility.
"""

import sys
import numpy as np
import sounddevice as sd
import threading

from modules.config import Config


class AudioProcessor:
    """Handles continuous audio capture from microphone with sample rate conversion."""
    
    # Maximum recording duration in seconds
    MAX_RECORDING_DURATION = 40.0
    
    def __init__(self, sample_rate=Config.SAMPLE_RATE, block_duration=Config.BLOCK_DURATION):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate for Whisper (16000 Hz)
            block_duration: Default block duration (used for display purposes)
        """
        self.target_sample_rate = sample_rate  # 16000 Hz for Whisper
        self.block_duration = block_duration
        self.is_recording = False
        self.device_id = None
        self.device_sample_rate = None  # Will be set to device's native rate
        
        # Continuous recording buffers
        self._audio_buffer = []
        self._recording_thread = None
        self._stop_recording_flag = False
        self._recording_start_time = None
        
        # Lock for thread-safe buffer access
        self._buffer_lock = threading.Lock()
        
    def _get_device_sample_rate(self):
        """Get the sample rate of the selected audio device."""
        try:
            if self.device_id is not None:
                device_info = sd.query_devices(self.device_id)
                return device_info['default_samplerate']
            else:
                # Use default device
                device_info = sd.query_devices()
                return device_info['default_samplerate']
        except Exception:
            # Default to 44100 Hz if we can't determine
            return 44100.0
    
    def _resample_audio(self, audio_data, orig_sr, target_sr):
        """
        Resample audio data from original sample rate to target sample rate.
        Uses simple linear interpolation for resampling.
        
        Args:
            audio_data: Input audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio array
        """
        if orig_sr == target_sr:
            return audio_data
        
        # Calculate the number of samples in the resampled audio
        num_samples = int(len(audio_data) * target_sr / orig_sr)
        
        # Create time indices for original and target
        orig_indices = np.arange(len(audio_data))
        target_indices = np.linspace(0, len(audio_data) - 1, num_samples)
        
        # Linear interpolation
        resampled = np.interp(target_indices, orig_indices, audio_data)
        return resampled.astype(np.float32)
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback function for continuous audio recording."""
        if status:
            print(f"Recording status: {status}")
        
        if self.is_recording:
            with self._buffer_lock:
                # Convert to float32 and normalize
                audio_chunk = indata.flatten().astype(np.float32) / 32768.0
                self._audio_buffer.append(audio_chunk.copy())
    
    def start_continuous_recording(self):
        """Start continuous audio recording."""
        try:
            # Get the device's native sample rate
            self.device_sample_rate = self._get_device_sample_rate()
            
            # Calculate block size for streaming (small blocks for responsive recording)
            block_size = int(self.device_sample_rate * 0.1)  # 100ms blocks
            
            print(f"  Using audio device {self.device_id}")
            print(f"  Device sample rate: {self.device_sample_rate} Hz")
            print(f"  Target sample rate: {self.target_sample_rate} Hz")
            print(f"  Max recording duration: {self.MAX_RECORDING_DURATION} seconds")
            self.is_recording = True
            self._audio_buffer = []
            self._recording_start_time = None
            print("✓ Continuous recording started - buffer cleared")
        except Exception as error:
            print(f"✗ Failed to start continuous recording: {error}")
            sys.exit(1)
    
    def stop_continuous_recording(self):
        """Stop continuous audio recording and return the recorded audio."""
        if self.is_recording:
            self.is_recording = False
            print("✓ Continuous recording stopped")
    
    def get_recorded_audio(self):
        """
        Get all recorded audio as a single array, resampled to target sample rate.
        
        Returns:
            Tuple of (resampled_audio_array, actual_duration_seconds)
        """
        with self._buffer_lock:
            if not self._audio_buffer:
                return np.array([], dtype=np.float32), 0.0
            
            # Concatenate all audio chunks
            full_audio = np.concatenate(self._audio_buffer)
            
            # Calculate actual duration at device sample rate
            actual_duration = len(full_audio) / self.device_sample_rate
            
            # Resample to target sample rate
            resampled_audio = self._resample_audio(
                full_audio,
                self.device_sample_rate,
                self.target_sample_rate
            )
            
            # Apply amplification
            resampled_audio = resampled_audio * Config.AMPLIFICATION_FACTOR
            resampled_audio = np.clip(resampled_audio, -1.0, 1.0)
            
            return resampled_audio, actual_duration
    
    def clear_buffer(self):
        """Clear the audio buffer."""
        with self._buffer_lock:
            self._audio_buffer = []
            self._recording_start_time = None
    
    def is_recording_active(self):
        """Check if recording is currently active."""
        return self.is_recording
    
    def get_buffer_duration(self):
        """Get the current duration of audio in the buffer (in seconds)."""
        with self._buffer_lock:
            if not self._audio_buffer:
                return 0.0
            
            total_samples = sum(len(chunk) for chunk in self._audio_buffer)
            return total_samples / self.device_sample_rate
    
    def start_recording(self):
        """Initialize audio recording system (legacy method)."""
        self.start_continuous_recording()
    
    def stop_recording(self):
        """Stop audio recording (legacy method)."""
        self.stop_continuous_recording()
    
    def _print_audio_stats(self, audio_data):
        """Print audio statistics for debugging."""
        max_amplitude = np.max(np.abs(audio_data))
        print(f"  Processed {len(audio_data)} samples, max amplitude: {max_amplitude:.6f}")
