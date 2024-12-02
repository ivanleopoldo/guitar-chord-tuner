import pyaudio
import sys

sys.path.append(".")
import queue
import threading
import librosa.feature
import numpy as np
from core import find_nearest, chordgram, chord_sequence
from core import frequency_array, frequency_to_note
import wave
import time
import tkinter as tk
from tkinter import ttk


def note_recognize():
    global current_note
    audio_data, sample_rate = librosa.load("tmp.wav")
    pitch_values = librosa.yin(audio_data, fmin=60, fmax=400)
    pitch_values[np.isnan(pitch_values)] = 0
    pitch_values = [
        find_nearest(frequency_array, pitch) for pitch in list(pitch_values)
    ]

    for pitch in pitch_values:
        latest_notes_list.append(float(pitch))
    if len(latest_notes_list) >= 5:
        max_note = max(latest_notes_list, key=latest_notes_list.count)
        if latest_notes_list.count(max_note) > 1:
            if max_note in frequency_array:
                possible_harmonic = frequency_array[
                    list(frequency_array).index(max_note) - 12
                ]
                global LOADING
                if LOADING:
                    print("complete!")
                    time.sleep(0.5)
                    LOADING = 0
                if possible_harmonic in latest_notes_list:
                    current_note = str(frequency_to_note[possible_harmonic])
                    print(current_note)
                else:
                    current_note = str(frequency_to_note[max_note])
                    print(current_note)
            else:
                current_note = "NN"
                print(current_note)
        else:
            current_note = "NN"
            print(current_note)
        latest_notes_list.clear()


def audio_callback(in_data, *args):
    q.put(in_data)
    ad_rdy_ev.set()
    return None, pyaudio.paContinue


def update_plot():
    global latest_notes_list, chord_set, current_note
    while not plot_queue.empty():
        latest_notes_list, chord_set, current_note = plot_queue.get()
    if chord_set:
        chord_label.config(text=f"Chord: {chord_set[0]}")
        note_label.config(text=f"Note: {current_note}")
    root.after(50, update_plot)  # Update more frequently


def read_audio_thread(audio_queue, audio_stream, audio_frames, audio_ready_event):
    global latest_notes_list, chord_set
    while audio_stream.is_active():
        audio_data = audio_queue.get()
        while not audio_queue.empty():
            audio_queue.get()

        wave_data = b"".join([audio_data])
        with wave.open("tmp.wav", "wb") as wave_file:
            wave_file.setnchannels(CHANNELS)
            wave_file.setsampwidth(pyaudio.get_sample_size(FORMAT))
            wave_file.setframerate(RATE)
            wave_file.writeframes(wave_data)

        note_recognize()
        chord_set = chord_sequence(
            chordgram("tmp.wav", sample_rate=44100, hop_length=4096)
        )
        if chord_set[0] == chord_set[1] == "NC":
            chord_set = ["NC"]
            print("NC")
        else:
            chord_set = [chord for chord in chord_set if chord != "NC"]
            for chord in chord_set:
                print(chord)
        plot_queue.put((latest_notes_list.copy(), chord_set.copy(), current_note))
        audio_ready_event.clear()


if __name__ == "__main__":
    LOADING = True
    CHUNK = 2048
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 22050
    latest_notes_list = [0] * 100  # Initialize with some values
    frames = []
    chord_set = []
    current_note = ""
    plot_queue = queue.Queue()

    p = pyaudio.PyAudio()
    q = queue.Queue()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=audio_callback,
        input_device_index=2,
    )

    stream.start_stream()

    ad_rdy_ev = threading.Event()
    thread_ = threading.Thread(
        target=read_audio_thread, args=(q, stream, frames, ad_rdy_ev)
    )
    thread_.daemon = True
    thread_.start()

    root = tk.Tk()
    root.title("Audio Visualization")

    chord_label = ttk.Label(root, text="Chord: ", font=("Helvetica", 16))
    chord_label.pack()

    note_label = ttk.Label(root, text="Note: ", font=("Helvetica", 16))
    note_label.pack()

    root.after(50, update_plot)  # Update more frequently
    root.mainloop()

    try:
        while stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    stream.stop_stream()
    stream.close()
    p.terminate()
