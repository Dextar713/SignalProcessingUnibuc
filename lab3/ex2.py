import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.artist as artist
from typing import List, Optional

fs = 100
f = 7
t = np.linspace(0, 1, fs)
signal0 = np.sin(2 * np.pi * f * t)

w = np.array([1, 3, 5, 7, 10])

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 5))
ax[0,0].plot(t, signal0)
ax[0,0].set_xlim(0, 1)

z = [0] * len(w)
scat_plots: List[Optional[artist.Artist]] = [None] * len(w)

for i in range(len(w)):
    z[i] = signal0 * np.exp(-2j * np.pi * w[i] * t)
    row, col = divmod(i + 1, 3)
    if i == 1:
        scat_plots[i] = ax[row, col].scatter([], [], s=20)
        ax[row, col].set_xlim(-1.5, 1.5)  # ← CRITICAL: Set limits
        ax[row, col].set_ylim(-1.5, 1.5)
    else:
        scat_plots[i] = ax[row, col].scatter(np.real(z[i]), np.imag(z[i]))
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


def update(frame):
    x = np.real(z[1][:frame+1])
    y = np.imag(z[1][:frame+1])
    data = np.column_stack([x, y])
    scat_plots[1].set_offsets(data)
    print(f"Frame {frame}: {len(x)} points, "
          f"first: ({x[0]:.3f}, {y[0]:.3f}), last: ({x[-1]:.3f}, {y[-1]:.3f})")
    return (scat_plots[1],)

anim = animation.FuncAnimation(fig=fig, func=update, frames=fs, interval=70, blit=False,
                               repeat=False)
# plt.savefig("ex2.pdf")
plt.show()


