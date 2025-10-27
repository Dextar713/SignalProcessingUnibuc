import scipy.signal as signal
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import  numpy as np
#import sounddevice as sd

A = 2
f = 5
fs = 10*f
fig, ax = plt.subplots(nrows=2, ncols=1)
time_axes = np.linspace(0, 1, fs)
sin_signal = np.sin(2*np.pi*f*time_axes)
cos_signal = np.cos(2*np.pi*f*time_axes+-np.pi/2)
ax[0].plot(time_axes, sin_signal)
ax[1].plot(time_axes, cos_signal)

plt.savefig('ex1.pdf')
plt.show()


