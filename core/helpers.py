import math
import librosa
import librosa.display
import numpy as np
from .constants import chord_template
import time


def find_nearest(array, value):
    if len(array) == 0:
        raise ValueError("Input array is empty")
    index = np.searchsorted(array, value, side="left")
    if index > 0 and (
        index == len(array)
        or math.fabs(value - array[index - 1]) < math.fabs(value - array[index])
    ):
        return array[index - 1]
    else:
        return array[index]


def cossim(vector_u, vector_v):
    return np.dot(vector_u, vector_v) / (
        np.linalg.norm(vector_u) * np.linalg.norm(vector_v)
    )


def chordgram(filename, sample_rate, hop_length):
    audio_data, _sample_rate = librosa.load(filename, sr=sample_rate)
    harmonic_audio, percussive_audio = librosa.effects.hpss(audio_data)
    start_time = time.time()
    chromagram = librosa.feature.chroma_cqt(
        y=harmonic_audio, sr=_sample_rate, hop_length=hop_length
    )
    num_frames = chromagram.shape[1]

    chords = list(chord_template.keys())
    chroma_vectors = np.transpose(chromagram)
    chordgram_matrix = []

    for frame in np.arange(num_frames):
        chroma_vector = chroma_vectors[frame]
        similarities = []

        for chord in chords:
            chord_vector = chord_template[chord]
            if chord == "NC":
                similarity = cossim(chroma_vector, chord_vector) * 0.8
            else:
                similarity = cossim(chroma_vector, chord_vector)
            similarities += [similarity]
        chordgram_matrix += [similarities]
    chordgram_matrix = np.transpose(chordgram_matrix)
    return np.array(chordgram_matrix)


def chord_sequence(chordgram_matrix):
    chords = list(chord_template.keys())
    num_frames = chordgram_matrix.shape[1]
    chordgram_matrix = np.transpose(chordgram_matrix)
    chord_sequence_list = []

    for frame in np.arange(num_frames):
        index = np.argmax(chordgram_matrix[frame])
        if chordgram_matrix[frame][index] == 0.0:
            chord = "NC"
        else:
            chord = chords[index]

        chord_sequence_list += [chord]
    return chord_sequence_list
