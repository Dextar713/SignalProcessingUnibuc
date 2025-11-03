import numpy as np
import matplotlib.pyplot as plt

fs = 50
t = np.linspace(0, 1, fs, endpoint=False)
f1, f2, f3 = 2, 5, 7
signal0 = ((np.sin(2 * np.pi * f1 * t) +
           0.5 * np.sin(2 * np.pi * f2 * t)) +
           2 * np.sin(2 * np.pi * f3 * t))

N = fs
F = np.zeros(shape=(N, N), dtype=complex)
for k in range(N):
    for n in range(N):
        F[k][n] = np.exp(-2j * np.pi * k * n / N)

spectrum0 = F@signal0

fig, axs = plt.subplots(1, 2, figsize=(12, 8))
axs[0].set_xlim(0, 1)
axs[0].plot(t, signal0)
freqs = np.linspace(0, fs, N, endpoint=False)
axs[1].set_xlim(freqs[0], freqs[-1])
axs[1].set_xticks(freqs[::2])
axs[1].stem(freqs, np.abs(spectrum0))
plt.tight_layout()
plt.savefig("ex3.pdf")
plt.show()

