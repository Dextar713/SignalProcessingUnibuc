import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.io.wavfile as wav

fs_play = 44100
t3 = np.linspace(0, 1, fs_play)
s3_sawtooth = lambda t : 2 * 1 * (240 * t - np.floor(240 * t + 0.5))

rate = int(10e5)
wav.write('sawtooth.wav', rate, s3_sawtooth(t3))
sawtooth_audio = wav.read('sawtooth.wav')[1]
sd.play(sawtooth_audio, fs_play)
sd.wait()