"""
Audio utility functions for preprocessing and feature extraction.
"""

import librosa
import numpy as np
import torch


def load_audio(file_path, sr=22050, duration=None):
    """
    Load audio file using librosa.
    
    Args:
        file_path (str): Path to audio file
        sr (int): Target sample rate
        duration (float): Duration to load (None = full file)
    
    Returns:
        tuple: (audio_data, sample_rate)
    """
    audio, sample_rate = librosa.load(file_path, sr=sr, duration=duration)
    return audio, sample_rate


def audio_to_melspectrogram(audio, sr=22050, n_mels=128, n_fft=2048, 
                            hop_length=512, duration=3.0):
    """
    Convert audio to mel spectrogram.
    
    Args:
        audio (np.array): Audio time series
        sr (int): Sample rate
        n_mels (int): Number of mel bands
        n_fft (int): FFT window size
        hop_length (int): Number of samples between frames
        duration (float): Fixed duration in seconds
    
    Returns:
        np.array: Mel spectrogram of shape (n_mels, time_steps)
    """
    # Ensure fixed length
    target_length = int(sr * duration)
    if len(audio) < target_length:
        # Pad if too short
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    else:
        # Trim if too long
        audio = audio[:target_length]
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm


def preprocess_audio_for_model(audio_path, sr=22050, n_mels=128, duration=3.0):
    """
    Complete preprocessing pipeline for model input.
    
    Args:
        audio_path (str): Path to audio file
        sr (int): Sample rate
        n_mels (int): Number of mel bands
        duration (float): Fixed duration
    
    Returns:
        torch.Tensor: Preprocessed mel spectrogram of shape (1, n_mels, time_steps)
    """
    # Load audio
    audio, _ = load_audio(audio_path, sr=sr, duration=duration)
    
    # Convert to mel spectrogram
    mel_spec = audio_to_melspectrogram(audio, sr=sr, n_mels=n_mels, duration=duration)
    
    # Add channel dimension and convert to tensor
    mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)  # (1, n_mels, time_steps)
    
    return mel_spec_tensor


def extract_features(audio, sr=22050):
    """
    Extract multiple audio features for analysis.
    
    Args:
        audio (np.array): Audio time series
        sr (int): Sample rate
    
    Returns:
        dict: Dictionary of extracted features
    """
    features = {}
    
    # Mel spectrogram
    features['mel_spectrogram'] = librosa.feature.melspectrogram(y=audio, sr=sr)
    
    # MFCCs
    features['mfcc'] = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Chroma features
    features['chroma'] = librosa.feature.chroma_stft(y=audio, sr=sr)
    
    # Spectral features
    features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)
    
    return features
