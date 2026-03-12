#!/usr/bin/env python3
"""
Real-time dictation module using Voice Activity Detection (VAD).

This module provides hands-free dictation by automatically detecting speech
using WebRTC VAD and transcribing with Whisper.

Based on the pywhispercpp Assistant pattern.
"""

import queue
import time
import logging
import sys
from typing import Callable, Optional
import numpy as np
import sounddevice as sd

from pywhispercpp.model import Model
import pywhispercpp.constants as constants


class RealtimeDictation:
    """
    Real-time dictation using Voice Activity Detection.
    
    Automatically detects speech and transcribes it without requiring
    a hotkey to be held. Perfect for hands-free dictation.
    
    Example usage:
        def on_text(text):
            print(f"Transcribed: {text}")
        
        dictation = RealtimeDictation(
            model_path='./models/ggml-base.bin',
            text_callback=on_text,
            language='sr'
        )
        dictation.start()
    """
    
    def __init__(
        self,
        model_path: str,
        text_callback: Callable[[str], None],
        language: str = 'sr',
        input_device: Optional[int] = None,
        silence_threshold: int = 8,
        queue_threshold: int = 16,
        block_duration_ms: int = 30,
        vad_aggressiveness: int = 1,
        n_threads: int = 8,
        print_realtime: bool = False,
        print_progress: bool = False,
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float = 0.0
    ):
        """
        Initialize real-time dictation.
        
        Args:
            model_path: Path to the Whisper GGML model
            text_callback: Function called with transcribed text
            language: Language code (e.g., 'sr' for Serbian, 'en' for English)
            input_device: Audio input device ID (None for default)
            silence_threshold: Frames of silence before transcribing
            queue_threshold: Minimum queue size before transcribing
            block_duration_ms: Audio block duration in milliseconds
            vad_aggressiveness: VAD aggressiveness (0-3, higher = more strict)
            n_threads: Number of threads for Whisper inference
            print_realtime: Print transcription in real-time
            print_progress: Print progress information
            beam_size: Beam size for beam search (higher = better quality, slower)
            best_of: Number of candidates for sampling (higher = better quality)
            temperature: Sampling temperature (0 = greedy, higher = more random)
        """
        self.model_path = model_path
        self.text_callback = text_callback
        self.language = language
        self.input_device = input_device
        self.silence_threshold = silence_threshold
        self.queue_threshold = queue_threshold
        self.block_duration_ms = block_duration_ms
        self.n_threads = n_threads
        self.print_realtime = print_realtime
        self.print_progress = print_progress
        self.beam_size = beam_size
        self.best_of = best_of
        self.temperature = temperature
        
        # Audio settings
        self.sample_rate = constants.WHISPER_SAMPLE_RATE  # 16000 Hz
        self.channels = 1
        self.block_size = int(self.sample_rate * block_duration_ms / 1000)
        
        # VAD setup - lazy import
        try:
            import webrtcvad
        except ImportError:
            print("Error: webrtcvad not installed. Run: pip install webrtcvad")
            sys.exit(1)
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.vad_aggressiveness = vad_aggressiveness  # Store for logging
        
        # Audio queue for buffering
        self.audio_queue = queue.Queue()
        
        # State
        self._silence_counter = 0
        self._is_running = False
        self._audio_stream = None
        
        # Initialize Whisper model
        logging.info(f"Loading Whisper model from {model_path}...")
        self.model = Model(
            model_path,
            n_threads=n_threads,
            print_realtime=print_realtime,
            print_progress=print_progress,
            print_timestamps=False,
            single_segment=True,
            no_context=True,
            language=language
        )
        logging.info("Model loaded successfully")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """
        Called for each audio block from sounddevice.
        
        Uses VAD to detect speech and buffers audio for transcription.
        """
        if status:
            logging.warning(f"Audio callback status: {status}")
        
        # Convert float32 [-1, 1] to int16 for VAD
        # VAD expects 16-bit PCM
        audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
        
        # Check if speech is detected
        is_speech = self.vad.is_speech(audio_int16.tobytes(), self.sample_rate)
        
        if is_speech:
            # Speech detected - add to queue and reset silence counter
            self.audio_queue.put(indata.copy())
            self._silence_counter = 0
            if self.print_progress:
                logging.debug("Speech detected")
        else:
            # Silence detected
            if self._silence_counter >= self.silence_threshold:
                # Enough silence - check if we have audio to transcribe
                if self.audio_queue.qsize() >= self.queue_threshold:
                    self._transcribe_speech()
                self._silence_counter = 0
            else:
                self._silence_counter += 1
    
    def _transcribe_speech(self):
        """
        Transcribe accumulated audio from the queue.
        
        Called when silence is detected after speech.
        """
        if self.print_progress:
            logging.info("Transcribing speech...")
        
        # Collect all audio from queue
        audio_chunks = []
        while not self.audio_queue.empty():
            try:
                audio_chunks.append(self.audio_queue.get_nowait())
            except queue.Empty:
                break
        
        if not audio_chunks:
            return
        
        # Concatenate audio chunks
        audio_data = np.concatenate(audio_chunks).flatten()
        
        # Add padding for small audio packets (helps with short commands)
        min_samples = int(self.sample_rate * 0.5)  # 0.5 seconds minimum
        if len(audio_data) < min_samples:
            padding = np.zeros(min_samples - len(audio_data))
            audio_data = np.concatenate([audio_data, padding])
        
        # Transcribe
        try:
            # The callback is already called during transcribe via new_segment_callback
            # Don't call it again after to avoid duplicates
            # Build params dict according to pywhispercpp PARAMS_SCHEMA
            params = {
                'temperature': self.temperature,
                'greedy': {'best_of': self.best_of},
                'beam_search': {'beam_size': self.beam_size, 'patience': -1.0}
            }
            self.model.transcribe(
                audio_data,
                new_segment_callback=self._on_new_segment,
                **params
            )
                    
        except Exception as e:
            logging.error(f"Transcription error: {e}")
    
    def _on_new_segment(self, segment):
        """
        Called when a new transcription segment is available.
        
        Args:
            segment: Segment object with text, t0, t1 attributes
        """
        text = segment.text.strip() if segment.text else ""
        
        if text and len(text) > 2:
            if self.print_progress:
                logging.info(f"Recognized: {text}")
            
            # Call user's callback
            if self.text_callback:
                self.text_callback(text)
    
    def start(self):
        """
        Start real-time dictation.
        
        Begins listening for speech and transcribing automatically.
        Runs until stop() is called or KeyboardInterrupt.
        """
        if self._is_running:
            logging.warning("RealtimeDictation is already running")
            return
        
        self._is_running = True
        
        logging.info("Starting real-time dictation...")
        logging.info(f"Language: {self.language}")
        logging.info(f"VAD aggressiveness: {self.vad_aggressiveness}")
        logging.info("Listening for speech... (Press Ctrl+C to stop)")
        
        try:
            with sd.InputStream(
                device=self.input_device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                dtype=np.float32,
                callback=self._audio_callback
            ) as self._audio_stream:
                
                # Keep the main thread alive
                while self._is_running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            logging.info("Stopped by user")
        except Exception as e:
            logging.error(f"Error in audio stream: {e}")
        finally:
            self._is_running = False
            self._audio_stream = None
            logging.info("Real-time dictation stopped")
    
    def stop(self):
        """
        Stop real-time dictation.
        
        Stops the audio stream and ends the listening loop.
        """
        self._is_running = False
        if self._audio_stream:
            try:
                self._audio_stream.stop()
            except Exception as e:
                logging.error(f"Error stopping stream: {e}")
    
    @staticmethod
    def list_audio_devices():
        """
        List available audio input devices.
        
        Returns:
            List of device information dictionaries
        """
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })
        
        return input_devices
