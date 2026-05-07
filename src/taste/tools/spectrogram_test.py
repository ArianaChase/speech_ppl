import os
import librosa
import librosa.display
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt

original_file = '/home/u5504709/new_work/speech_ppl/speechocean762/WAVE/SPEAKER0001/000010011.WAV'  
reconstructed_file = "/home/u5504709/new_work/speech_ppl/work/outputs/reconstructed.wav"

# load audio files with librosa
original, sr = librosa.load(original_file)
reconstructed, sr = librosa.load(reconstructed_file)

# short-time fourier transform
FRAME_SIZE = 2048
HOP_SIZE = 512

S_original = librosa.stft(original, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
Y_original = np.abs(S_original) ** 2

def plot_spectrogram(Y, sr, hop_length, y_axis="log"):
    plt.figure(figsize=(25,10))
    librosa.display.specshow(Y,
                             sr = sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")

plot_spectrogram(Y_original, sr, HOP_SIZE)

plt.savefig("work/outputs/spectrogram.png")