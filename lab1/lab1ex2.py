import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8), nrows=4, ncols=1)

t1 = np.linspace(0, 1/400, 1600)
s1 = lambda t: np.sin(400*2*np.pi*t)

t2 = np.linspace(0, 3, 1600*32)
s2 = lambda t: np.sin(800*2*np.pi*t)

t3 = np.linspace(0, 1, 240 * 10)
s3_sawtooth = lambda t : 2 * 1 * (240 * t - np.floor(240 * t + 0.5))

t4 = np.linspace(0, 1, 300 * 20)
s4 = lambda t: np.sign(np.sin(300*2*np.pi*t))

ax[0].set_xlim([0, 1/400])
ax[0].plot(t1, s1(t1))

ax[1].set_xlim(0, 1/400)
ax[1].plot(t2, s2(t2))

ax[2].set_xlim([0, 0.05])
ax[2].plot(t3, s3_sawtooth(t3))

ax[3].set_xlim([0, 0.02])
ax[3].plot(t4, s4(t4))

plt.savefig("Lab1ex2.pdf")
plt.show()