import numpy as np
import matplotlib.pyplot as plt

f = 4
fs = 10*f
A = 1
n_phases = 4
phases = np.linspace(0, 2*np.pi, 4, endpoint=False)
# print(phases)
time_axes = np.linspace(0, 1, fs)
s = [0]*n_phases
for i in range(len(phases)):
    s[i] = A*np.sin(2*np.pi*f*time_axes+phases[i])

SNR = [0.1, 1, 10, 100]

fig, ax = plt.subplots(nrows=4, ncols=2)
plt.tight_layout()
for i in range(len(phases)):
    z_noise = np.random.normal(size=len(time_axes))
    y_param = np.sqrt(np.linalg.norm(s[i]) / (np.linalg.norm(z_noise) * SNR[2]))
    ax[i][0].set_xlim(0, 1)
    ax[i][0].plot(time_axes, s[i])
    ax[i][0].hlines(0, time_axes[0], time_axes[-1], colors='r')
    ax[i][1].set_xlim(0, 1)
    ax[i][1].plot(time_axes, s[i] + y_param*z_noise)
    ax[i][1].hlines(0, time_axes[0], time_axes[-1], colors='r')

plt.savefig('ex2.pdf')
plt.show()

