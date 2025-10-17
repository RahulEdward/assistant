"""
Audio Utilities
Supporting classes for audio processing, noise reduction, and voice management.
"""

import numpy as np
import scipy.signal
import scipy.fft
from typing import Dict, List, Optional, Tuple
import logging
import json
from pathlib import Path


class AudioPostprocessor:
    """Audio post-processing utilities"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
    
    def normalize_audio(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normalize audio to target dB level"""
        try:
            # Calculate RMS
            rms = np.sqrt(np.mean(audio ** 2))
            
            if rms > 0:
                # Convert target dB to linear scale
                target_rms = 10 ** (target_db / 20.0)
                
                # Apply normalization
                normalized = audio * (target_rms / rms)
                
                # Prevent clipping
                max_val = np.max(np.abs(normalized))
                if max_val > 1.0:
                    normalized = normalized / max_val
                
                return normalized
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Audio normalization error: {e}")
            return audio
    
    def apply_fade(self, audio: np.ndarray, fade_in_ms: int = 50, fade_out_ms: int = 50) -> np.ndarray:
        """Apply fade in/out to audio"""
        try:
            fade_in_samples = int(fade_in_ms * self.sample_rate / 1000)
            fade_out_samples = int(fade_out_ms * self.sample_rate / 1000)
            
            result = audio.copy()
            
            # Fade in
            if fade_in_samples > 0 and len(result) > fade_in_samples:
                fade_in = np.linspace(0, 1, fade_in_samples)
                result[:fade_in_samples] *= fade_in
            
            # Fade out
            if fade_out_samples > 0 and len(result) > fade_out_samples:
                fade_out = np.linspace(1, 0, fade_out_samples)
                result[-fade_out_samples:] *= fade_out
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fade application error: {e}")
            return audio
    
    def remove_silence(self, audio: np.ndarray, threshold: float = 0.01, 
                      min_silence_ms: int = 100) -> np.ndarray:
        """Remove silence from audio"""
        try:
            min_silence_samples = int(min_silence_ms * self.sample_rate / 1000)
            
            # Find non-silent regions
            energy = np.abs(audio)
            is_speech = energy > threshold
            
            # Apply morphological operations to clean up detection
            kernel_size = min(min_silence_samples, len(audio) // 10)
            if kernel_size > 0:
                kernel = np.ones(kernel_size)
                is_speech = np.convolve(is_speech.astype(float), kernel, mode='same') > 0.5
            
            # Find start and end of speech
            speech_indices = np.where(is_speech)[0]
            
            if len(speech_indices) > 0:
                start_idx = speech_indices[0]
                end_idx = speech_indices[-1] + 1
                return audio[start_idx:end_idx]
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Silence removal error: {e}")
            return audio


class NoiseReducer:
    """Advanced noise reduction for audio"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        self.noise_profile: Optional[np.ndarray] = None
    
    def estimate_noise_profile(self, audio: np.ndarray, noise_duration_ms: int = 500) -> bool:
        """Estimate noise profile from beginning of audio"""
        try:
            noise_samples = int(noise_duration_ms * self.sample_rate / 1000)
            
            if len(audio) < noise_samples:
                noise_samples = len(audio) // 4
            
            noise_segment = audio[:noise_samples]
            
            # Compute power spectral density
            freqs, psd = scipy.signal.welch(
                noise_segment, 
                fs=self.sample_rate,
                nperseg=min(1024, len(noise_segment))
            )
            
            self.noise_profile = psd
            return True
            
        except Exception as e:
            self.logger.error(f"Noise profile estimation error: {e}")
            return False
    
    def reduce_noise(self, audio: np.ndarray, reduction_factor: float = 0.8) -> np.ndarray:
        """Apply spectral subtraction noise reduction"""
        try:
            if self.noise_profile is None:
                # Estimate noise profile if not available
                self.estimate_noise_profile(audio)
            
            # STFT
            f, t, stft = scipy.signal.stft(
                audio, 
                fs=self.sample_rate,
                nperseg=1024,
                noverlap=512
            )
            
            # Magnitude and phase
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Spectral subtraction
            if self.noise_profile is not None:
                # Interpolate noise profile to match STFT frequencies
                noise_interp = np.interp(f, 
                                       np.linspace(0, self.sample_rate/2, len(self.noise_profile)),
                                       self.noise_profile)
                
                # Apply spectral subtraction
                noise_magnitude = np.sqrt(noise_interp)[:, np.newaxis]
                
                # Subtract noise with over-subtraction factor
                clean_magnitude = magnitude - reduction_factor * noise_magnitude
                
                # Apply spectral floor (prevent over-subtraction)
                spectral_floor = 0.1 * magnitude
                clean_magnitude = np.maximum(clean_magnitude, spectral_floor)
            else:
                clean_magnitude = magnitude
            
            # Reconstruct signal
            clean_stft = clean_magnitude * np.exp(1j * phase)
            
            # ISTFT
            _, clean_audio = scipy.signal.istft(
                clean_stft,
                fs=self.sample_rate,
                nperseg=1024,
                noverlap=512
            )
            
            return clean_audio.astype(audio.dtype)
            
        except Exception as e:
            self.logger.error(f"Noise reduction error: {e}")
            return audio


class VoiceActivityDetector:
    """Voice Activity Detection using energy and spectral features"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        
        # VAD parameters
        self.frame_length_ms = 25
        self.frame_shift_ms = 10
        self.energy_threshold = 0.01
        self.spectral_centroid_threshold = 1000
    
    def detect_voice_activity(self, audio: np.ndarray) -> np.ndarray:
        """Detect voice activity in audio signal"""
        try:
            frame_length = int(self.frame_length_ms * self.sample_rate / 1000)
            frame_shift = int(self.frame_shift_ms * self.sample_rate / 1000)
            
            num_frames = (len(audio) - frame_length) // frame_shift + 1
            vad_decisions = np.zeros(num_frames, dtype=bool)
            
            for i in range(num_frames):
                start_idx = i * frame_shift
                end_idx = start_idx + frame_length
                frame = audio[start_idx:end_idx]
                
                # Energy-based detection
                energy = np.sum(frame ** 2) / len(frame)
                energy_decision = energy > self.energy_threshold
                
                # Spectral centroid-based detection
                spectral_centroid = self._compute_spectral_centroid(frame)
                spectral_decision = spectral_centroid > self.spectral_centroid_threshold
                
                # Combine decisions
                vad_decisions[i] = energy_decision and spectral_decision
            
            # Smooth VAD decisions
            vad_decisions = self._smooth_vad_decisions(vad_decisions)
            
            return vad_decisions
            
        except Exception as e:
            self.logger.error(f"VAD error: {e}")
            return np.ones(len(audio) // (frame_shift or 1), dtype=bool)
    
    def _compute_spectral_centroid(self, frame: np.ndarray) -> float:
        """Compute spectral centroid of audio frame"""
        try:
            # Compute FFT
            fft = np.fft.rfft(frame)
            magnitude = np.abs(fft)
            
            # Frequency bins
            freqs = np.fft.rfftfreq(len(frame), 1/self.sample_rate)
            
            # Spectral centroid
            if np.sum(magnitude) > 0:
                centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            else:
                centroid = 0
            
            return centroid
            
        except Exception as e:
            return 0
    
    def _smooth_vad_decisions(self, vad_decisions: np.ndarray, 
                            min_speech_frames: int = 5, 
                            min_silence_frames: int = 3) -> np.ndarray:
        """Smooth VAD decisions to remove spurious detections"""
        try:
            smoothed = vad_decisions.copy()
            
            # Remove short speech segments
            speech_segments = self._find_segments(vad_decisions, True)
            for start, end in speech_segments:
                if end - start < min_speech_frames:
                    smoothed[start:end] = False
            
            # Remove short silence segments
            silence_segments = self._find_segments(smoothed, False)
            for start, end in silence_segments:
                if end - start < min_silence_frames:
                    smoothed[start:end] = True
            
            return smoothed
            
        except Exception as e:
            return vad_decisions
    
    def _find_segments(self, decisions: np.ndarray, value: bool) -> List[Tuple[int, int]]:
        """Find segments with specific value"""
        segments = []
        in_segment = False
        start_idx = 0
        
        for i, decision in enumerate(decisions):
            if decision == value and not in_segment:
                start_idx = i
                in_segment = True
            elif decision != value and in_segment:
                segments.append((start_idx, i))
                in_segment = False
        
        # Handle case where segment extends to end
        if in_segment:
            segments.append((start_idx, len(decisions)))
        
        return segments


class VoiceProfileManager:
    """Manage voice profiles and characteristics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.profiles: Dict[str, Dict] = {}
    
    def create_profile(self, profile_id: str, characteristics: Dict) -> bool:
        """Create a new voice profile"""
        try:
            # Validate characteristics
            required_fields = ['pitch_scale', 'speed_scale', 'energy_scale']
            
            for field in required_fields:
                if field not in characteristics:
                    self.logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate ranges
            if not (0.5 <= characteristics['pitch_scale'] <= 2.0):
                self.logger.error("Pitch scale must be between 0.5 and 2.0")
                return False
            
            if not (0.5 <= characteristics['speed_scale'] <= 2.0):
                self.logger.error("Speed scale must be between 0.5 and 2.0")
                return False
            
            if not (0.1 <= characteristics['energy_scale'] <= 3.0):
                self.logger.error("Energy scale must be between 0.1 and 3.0")
                return False
            
            self.profiles[profile_id] = characteristics
            return True
            
        except Exception as e:
            self.logger.error(f"Profile creation error: {e}")
            return False
    
    def get_profile(self, profile_id: str) -> Optional[Dict]:
        """Get voice profile by ID"""
        return self.profiles.get(profile_id)
    
    def list_profiles(self) -> List[str]:
        """List all available profile IDs"""
        return list(self.profiles.keys())
    
    def update_profile(self, profile_id: str, updates: Dict) -> bool:
        """Update existing voice profile"""
        try:
            if profile_id not in self.profiles:
                self.logger.error(f"Profile not found: {profile_id}")
                return False
            
            # Update profile
            self.profiles[profile_id].update(updates)
            return True
            
        except Exception as e:
            self.logger.error(f"Profile update error: {e}")
            return False
    
    def delete_profile(self, profile_id: str) -> bool:
        """Delete voice profile"""
        try:
            if profile_id in self.profiles:
                del self.profiles[profile_id]
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Profile deletion error: {e}")
            return False


class AudioBuffer:
    """Circular audio buffer for real-time processing"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = np.zeros(max_size, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.size = 0
        self.logger = logging.getLogger(__name__)
    
    def write(self, data: np.ndarray) -> int:
        """Write data to buffer, returns number of samples written"""
        try:
            data = data.astype(np.float32)
            samples_to_write = min(len(data), self.max_size - self.size)
            
            if samples_to_write <= 0:
                return 0
            
            # Handle wrap-around
            if self.write_pos + samples_to_write <= self.max_size:
                self.buffer[self.write_pos:self.write_pos + samples_to_write] = data[:samples_to_write]
            else:
                # Split write
                first_part = self.max_size - self.write_pos
                self.buffer[self.write_pos:] = data[:first_part]
                self.buffer[:samples_to_write - first_part] = data[first_part:samples_to_write]
            
            self.write_pos = (self.write_pos + samples_to_write) % self.max_size
            self.size += samples_to_write
            
            return samples_to_write
            
        except Exception as e:
            self.logger.error(f"Buffer write error: {e}")
            return 0
    
    def read(self, num_samples: int) -> np.ndarray:
        """Read data from buffer"""
        try:
            samples_to_read = min(num_samples, self.size)
            
            if samples_to_read <= 0:
                return np.array([], dtype=np.float32)
            
            result = np.zeros(samples_to_read, dtype=np.float32)
            
            # Handle wrap-around
            if self.read_pos + samples_to_read <= self.max_size:
                result = self.buffer[self.read_pos:self.read_pos + samples_to_read].copy()
            else:
                # Split read
                first_part = self.max_size - self.read_pos
                result[:first_part] = self.buffer[self.read_pos:]
                result[first_part:] = self.buffer[:samples_to_read - first_part]
            
            self.read_pos = (self.read_pos + samples_to_read) % self.max_size
            self.size -= samples_to_read
            
            return result
            
        except Exception as e:
            self.logger.error(f"Buffer read error: {e}")
            return np.array([], dtype=np.float32)
    
    def peek(self, num_samples: int) -> np.ndarray:
        """Peek at data without removing from buffer"""
        try:
            samples_to_peek = min(num_samples, self.size)
            
            if samples_to_peek <= 0:
                return np.array([], dtype=np.float32)
            
            result = np.zeros(samples_to_peek, dtype=np.float32)
            
            # Handle wrap-around
            if self.read_pos + samples_to_peek <= self.max_size:
                result = self.buffer[self.read_pos:self.read_pos + samples_to_peek].copy()
            else:
                # Split peek
                first_part = self.max_size - self.read_pos
                result[:first_part] = self.buffer[self.read_pos:]
                result[first_part:] = self.buffer[:samples_to_peek - first_part]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Buffer peek error: {e}")
            return np.array([], dtype=np.float32)
    
    def available_space(self) -> int:
        """Get available space in buffer"""
        return self.max_size - self.size
    
    def available_data(self) -> int:
        """Get available data in buffer"""
        return self.size
    
    def clear(self):
        """Clear buffer"""
        self.write_pos = 0
        self.read_pos = 0
        self.size = 0
        self.buffer.fill(0)