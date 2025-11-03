import numpy as np
import matplotlib.pyplot as plt

fs = 100
f = 7
t = np.linspace(0, 1, fs)
signal0 = np.sin(2 * np.pi * f * t)

w = np.array([1, 3, 5, 7, 10])

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 5))
ax[0,0].plot(t, signal0)
ax[0,0].set_xlim(0, 1)

z = [0] * len(w)
for i in range(len(w)):
    z[i] = signal0 * np.exp(-2j * np.pi * w[i] * t)
    row, col = divmod(i + 1, 3)
    ax[row, col].scatter(np.real(z[i]), np.imag(z[i]))
    ax[row, col].set_aspect('equal')
    ax[row, col].set_title(f"Frequency {w[i]} Hz")

time_sample = 17
ax[0, 0].stem(t[time_sample], signal0[time_sample], 'r')
ax[0, 0].hlines(0, t[0], t[-1], color='r')
ax[0, 0].set_xlim(0, 1)
complex_z = signal0[time_sample] * np.exp(-2j * np.pi*1*t[time_sample])
ax[0, 1].plot(np.real(complex_z), np.imag(complex_z), marker='o', markersize=10, color='r')
ax[0, 1].plot([np.real(complex_z), 0], [np.imag(complex_z), 0], color='r')
plt.tight_layout()
plt.savefig("ex2.pdf")
plt.show()


