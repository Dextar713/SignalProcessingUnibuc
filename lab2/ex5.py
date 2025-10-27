import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import sounddevice as sd

A = 1
fs_play = 44100
f1, f2 = 100, 300
fs1, fs2 = fs_play, fs_play
t1, t2 = (np.linspace(0, 1, fs1, endpoint=False),
          np.linspace(0, 1, fs2, endpoint=False))
sinus1 = np.sin(2*np.pi*f1*t1)
sinus2 = np.sin(2*np.pi*f2*t2)
t12 = np.append(t1, t2 + 1)
sinus12 = np.append(sinus1, sinus2)
plt.xlim(0, 20/f2)
plt.plot(t12, sinus12)
plt.savefig('ex5.pdf')
plt.show()

rate = int(10e5)
wav.write('sinus12.wav', rate, sinus12)
sd.play(sinus12, fs_play)
sd.wait()