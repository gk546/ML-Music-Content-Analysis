"""
Preprocessing functions for audio data.
"""

import numpy as np
import librosa
import traceback


def preprocess_audio(audio_data, sample_rate):
    """
    Preprocess audio data with normalization and filtering.

    Args:
        audio_data (np.ndarray): Raw audio time series
        sample_rate (int): Sample rate

    Returns:
        np.ndarray: Preprocessed audio
    """
    # Normalize audio to [-1, 1]
    audio_normalized = audio_data / (np.max(np.abs(audio_data)) + 1e-8)

    # Remove DC offset
    audio_normalized = audio_normalized - np.mean(audio_normalized)

    # Pre-emphasis filter
    pre_emphasis = 0.97
    audio_emphasized = np.append(
        audio_normalized[0],
        audio_normalized[1:] - pre_emphasis * audio_normalized[:-1]
    )

    return audio_emphasized


def augment_audio(audio, sr=22050):
    """
    Apply data augmentation techniques to audio.
    """
    augmentation_type = np.random.choice(['original', 'pitch', 'time', 'noise'])

    if augmentation_type == 'pitch':
        n_steps = np.random.randint(-2, 3)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    elif augmentation_type == 'time':
        rate = np.random.uniform(0.9, 1.1)
        audio = librosa.effects.time_stretch(audio, rate=rate)

    elif augmentation_type == 'noise':
        noise = np.random.randn(len(audio)) * 0.005
        audio = audio + noise

    return audio


# ---------------- TEST BLOCK ----------------
if __name__ == "__main__":
    import os

    TEST_AUDIO = os.path.join(
        "datasets", "chords", "C",
        "y2mate.com - Bleach OST  Number One Instrumental HD_480p.wav"
    )

    print("Testing preprocessing on:", TEST_AUDIO)

    try:
        # LOAD AUDIO FIRST (this was missing before)
        audio, sr = librosa.load(TEST_AUDIO, sr=22050, mono=True)

        print("Raw audio shape:", audio.shape)
        print("Sample rate:", sr)

        processed_audio = preprocess_audio(audio, sr)

        print("Processed audio shape:", processed_audio.shape)
        print("Processed audio dtype:", processed_audio.dtype)

    except Exception:
        print("ERROR OCCURRED:")
        traceback.print_exc()

