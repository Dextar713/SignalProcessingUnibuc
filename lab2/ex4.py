import numpy as np
import matplotlib.pyplot as plt

A = 1
f = 3
fs = 100*f
time_axes = np.linspace(0, 1, fs)
s_sinus = A*np.sin(2*np.pi*f*time_axes)
s_sawtooth = 2 * A * (f * time_axes - np.floor(f * time_axes + 0.5))
fig, ax = plt.subplots(nrows=3, ncols=1)
ax[0].plot(time_axes, s_sinus)
ax[1].plot(time_axes, s_sawtooth)
ax[2].plot(time_axes, s_sinus + s_sawtooth)
plt.savefig('ex4.pdf')
plt.show()
