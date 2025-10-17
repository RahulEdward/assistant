"""
Voice Processor - Custom STT/TTS Engine
Handles voice input/output with proprietary speech recognition and synthesis engines.
Designed for high accuracy (≥95%) and sub-second response times.
"""

import asyncio
import logging
import numpy as np
import pyaudio
import wave
import threading
import queue
from typing import Optional, Callable, Dict, Any
from datetime import datetime
import io

from .stt_engine import CustomSTTEngine
from .tts_engine import CustomTTSEngine
from .audio_utils import AudioProcessor, NoiseReducer, VoiceActivityDetector


class VoiceProcessor:
    """Main voice processing coordinator"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Audio configuration
        self.sample_rate = config.sample_rate
        self.channels = config.channels
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        
        # Components
        self.stt_engine: Optional[CustomSTTEngine] = None
        self.tts_engine: Optional[CustomTTSEngine] = None
        self.audio_processor: Optional[AudioProcessor] = None
        self.noise_reducer: Optional[NoiseReducer] = None
        self.vad: Optional[VoiceActivityDetector] = None
        
        # Audio stream
        self.audio = None
        self.input_stream = None
        self.output_stream = None
        
        # State management
        self.is_listening = False
        self.is_speaking = False
        self.audio_queue = queue.Queue()
        self.callback_function: Optional[Callable] = None
        
        # Performance metrics
        self.recognition_accuracy = 0.0
        self.response_times = []
    
    async def initialize(self):
        """Initialize all voice processing components"""
        try:
            self.logger.info("Initializing Voice Processor...")
            
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()
            
            # Initialize custom engines
            self.stt_engine = CustomSTTEngine(self.config)
            await self.stt_engine.initialize()
            
            self.tts_engine = CustomTTSEngine(self.config)
            await self.tts_engine.initialize()
            
            # Initialize audio processing components
            self.audio_processor = AudioProcessor(self.sample_rate, self.channels)
            self.noise_reducer = NoiseReducer(self.sample_rate)
            self.vad = VoiceActivityDetector(
                self.sample_rate,
                threshold=self.config.voice_activation_threshold
            )
            
            # Test audio devices
            await self._test_audio_devices()
            
            self.logger.info("Voice Processor initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Voice Processor: {e}")
            return False
    
    async def _test_audio_devices(self):
        """Test audio input/output devices"""
        try:
            # Get default input device
            default_input = self.audio.get_default_input_device_info()
            self.logger.info(f"Default input device: {default_input['name']}")
            
            # Get default output device
            default_output = self.audio.get_default_output_device_info()
            self.logger.info(f"Default output device: {default_output['name']}")
            
            # Test input stream
            test_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            test_stream.close()
            
        except Exception as e:
            self.logger.warning(f"Audio device test failed: {e}")
    
    async def start_listening(self, callback: Callable):
        """Start continuous voice listening"""
        if self.is_listening:
            self.logger.warning("Already listening")
            return
        
        self.callback_function = callback
        self.is_listening = True
        
        try:
            # Open input stream
            self.input_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.input_stream.start_stream()
            
            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._audio_processing_loop,
                daemon=True
            )
            self.processing_thread.start()
            
            self.logger.info("Started voice listening")
            
        except Exception as e:
            self.logger.error(f"Failed to start listening: {e}")
            self.is_listening = False
            raise
    
    async def stop_listening(self):
        """Stop voice listening"""
        self.is_listening = False
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
        
        self.logger.info("Stopped voice listening")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for incoming audio data"""
        if self.is_listening and not self.is_speaking:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def _audio_processing_loop(self):
        """Background thread for processing audio data"""
        audio_buffer = []
        silence_counter = 0
        max_silence = int(self.sample_rate / self.chunk_size * 2)  # 2 seconds
        
        while self.is_listening:
            try:
                # Get audio data with timeout
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Apply noise reduction if enabled
                if self.config.noise_reduction and self.noise_reducer:
                    audio_array = self.noise_reducer.reduce_noise(audio_array)
                
                # Voice activity detection
                has_voice = self.vad.detect_voice(audio_array)
                
                if has_voice:
                    audio_buffer.append(audio_array)
                    silence_counter = 0
                else:
                    silence_counter += 1
                
                # Process accumulated audio when silence detected
                if silence_counter >= max_silence and audio_buffer:
                    combined_audio = np.concatenate(audio_buffer)
                    audio_buffer = []
                    silence_counter = 0
                    
                    # Process the audio asynchronously
                    asyncio.run_coroutine_threadsafe(
                        self._process_audio_chunk(combined_audio.tobytes()),
                        asyncio.get_event_loop()
                    )
                    
            except Exception as e:
                self.logger.error(f"Error in audio processing loop: {e}")
    
    async def _process_audio_chunk(self, audio_data: bytes):
        """Process a chunk of audio data"""
        try:
            if self.callback_function:
                await self.callback_function(audio_data)
        except Exception as e:
            self.logger.error(f"Error processing audio chunk: {e}")
    
    async def speech_to_text(self, audio_data: bytes) -> Optional[str]:
        """Convert speech to text using custom STT engine"""
        start_time = datetime.now()
        
        try:
            # Use custom STT engine
            text = await self.stt_engine.transcribe(audio_data)
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.response_times.append(processing_time)
            
            # Keep only last 100 response times
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]
            
            if text:
                self.logger.info(f"STT: '{text}' (processed in {processing_time:.3f}s)")
                return text.strip()
            
        except Exception as e:
            self.logger.error(f"STT error: {e}")
        
        return None
    
    async def text_to_speech(self, text: str, voice_id: str = "default") -> bool:
        """Convert text to speech using custom TTS engine"""
        try:
            self.is_speaking = True
            
            # Generate speech audio
            audio_data = await self.tts_engine.synthesize(text, voice_id)
            
            if audio_data:
                # Play the audio
                await self._play_audio(audio_data)
                self.logger.info(f"TTS: Spoke '{text[:50]}...'")
                return True
            
        except Exception as e:
            self.logger.error(f"TTS error: {e}")
        finally:
            self.is_speaking = False
        
        return False
    
    async def _play_audio(self, audio_data: bytes):
        """Play audio data through speakers"""
        try:
            # Open output stream
            output_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True
            )
            
            # Play audio in chunks
            chunk_size = 1024
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                output_stream.write(chunk)
            
            output_stream.close()
            
        except Exception as e:
            self.logger.error(f"Error playing audio: {e}")
    
    async def set_voice_activation_threshold(self, threshold: float):
        """Update voice activation threshold"""
        self.config.voice_activation_threshold = threshold
        if self.vad:
            self.vad.set_threshold(threshold)
        self.logger.info(f"Voice activation threshold set to {threshold}")
    
    async def calibrate_noise_level(self, duration: float = 3.0):
        """Calibrate background noise level"""
        try:
            self.logger.info(f"Calibrating noise level for {duration} seconds...")
            
            # Collect background noise samples
            noise_samples = []
            start_time = datetime.now()
            
            while (datetime.now() - start_time).total_seconds() < duration:
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    noise_samples.append(audio_array)
                except queue.Empty:
                    continue
            
            if noise_samples:
                # Calculate noise profile
                combined_noise = np.concatenate(noise_samples)
                if self.noise_reducer:
                    await self.noise_reducer.calibrate(combined_noise)
                
                # Update VAD threshold
                noise_level = np.std(combined_noise)
                new_threshold = max(0.1, min(0.9, noise_level / 32768.0 * 2))
                await self.set_voice_activation_threshold(new_threshold)
                
                self.logger.info("Noise calibration completed")
                return True
            
        except Exception as e:
            self.logger.error(f"Noise calibration error: {e}")
        
        return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get voice processing performance metrics"""
        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times else 0.0
        )
        
        return {
            'recognition_accuracy': self.recognition_accuracy,
            'average_response_time': avg_response_time,
            'total_recognitions': len(self.response_times),
            'is_listening': self.is_listening,
            'is_speaking': self.is_speaking
        }
    
    async def apply_optimization(self, action: str, parameters: Dict[str, Any]):
        """Apply performance optimization"""
        try:
            if action == "adjust_chunk_size":
                self.chunk_size = parameters.get("chunk_size", self.chunk_size)
            elif action == "update_threshold":
                threshold = parameters.get("threshold", self.config.voice_activation_threshold)
                await self.set_voice_activation_threshold(threshold)
            elif action == "toggle_noise_reduction":
                self.config.noise_reduction = parameters.get("enabled", True)
            
            self.logger.info(f"Applied optimization: {action}")
            
        except Exception as e:
            self.logger.error(f"Error applying optimization: {e}")
    
    async def cleanup(self):
        """Cleanup voice processor resources"""
        self.logger.info("Cleaning up Voice Processor...")
        
        await self.stop_listening()
        
        if self.audio:
            self.audio.terminate()
        
        if self.stt_engine:
            await self.stt_engine.cleanup()
        
        if self.tts_engine:
            await self.tts_engine.cleanup()
        
        self.logger.info("Voice Processor cleanup completed")