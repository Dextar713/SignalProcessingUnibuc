import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import sounddevice as sd

fs, vocals_audio_stereo = wav.read("ef11d21e.wav")
# shape = (N, 2) => stereo audio 2 channels left + right
# sd.play(vocals_audio, fs)
# sd.wait()

mono_audio = vocals_audio_stereo.mean(axis=1)
N = mono_audio.shape[0]
window_size = int(0.01 * N)
step_size = window_size // 2

window_groups = []
for i in range(0, N - window_size, step_size):
    window_groups.append(mono_audio[i:i + window_size])
window_groups = np.array(window_groups, dtype=np.float64)

spectrum = np.abs(np.fft.fft(window_groups, axis=1))
spectrum_db = 20 * np.log10(spectrum + 1e-6)

first_half_idx = window_size // 2
spectrum_plot = spectrum[:, :first_half_idx]

print(spectrum.shape)

time = np.arange(0, N - window_size, step_size, dtype=np.float64) / fs
freqs = np.arange(first_half_idx, dtype=np.float64) * fs / window_size
freqs_db = 20 * np.log10(freqs + 1e-6)
print(time[0])
plt.figure(figsize=(10, 6))
plt.imshow(spectrum_plot.T, cmap="hot", aspect="auto", origin="lower",
           extent=(time[0], time[-1], freqs[0], freqs[-1])
           )
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.title("Audio frequency spectrum")
plt.ylim(0, 5000)
plt.tight_layout()
plt.savefig("ex6.pdf")
plt.show()
