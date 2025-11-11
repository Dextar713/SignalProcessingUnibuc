import matplotlib.pyplot as plt
import numpy as np

f = 10
fs = 18
t = np.linspace(0, 1, fs, endpoint=False)
s = np.sin(2 * np.pi * f * t)
s1 = np.sin(2 * np.pi * (f + 1*fs) * t)
s2 = np.sin(2 * np.pi * (f + 2*fs) * t)
t_fin = np.linspace(0, 1, fs*100, endpoint=False)
s_fin = np.sin(2 * np.pi * f * t_fin)
s_fin_aliased = np.sin(2 * np.pi * np.abs(fs-f) * t_fin + np.pi)

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 5))
ax[0,0].set_ylim(-1, 1)
ax[1,0].set_ylim(-1, 1)
ax[2,0].set_ylim(-1, 1)
ax[0, 0].set_xlim(0, 1)
ax[1, 0].set_xlim(0, 1)
ax[2, 0].set_xlim(0, 1)
ax[0,0].stem(t, s)
ax[1,0].stem(t, s1)
ax[2,0].stem(t, s2)
ax[0, 0].plot(t_fin, s_fin_aliased, color='green')
ax[1, 0].plot(t_fin, s_fin_aliased, color='green')
ax[2, 0].plot(t_fin, s_fin_aliased, color='green')
ax[0,0].set_title(f'Frequency {f} Hz')
ax[1,0].set_title(f'Frequency {f+1*fs} Hz')
ax[2,0].set_title(f'Frequency {f+2*fs} Hz')
# ax[1].plot(t, s)

ax[0, 1].set_ylim(-1, 1)
ax[1, 1].set_ylim(-1, 1)
ax[2, 1].set_ylim(-1, 1)
ax[0, 1].set_xlim(0, 1)
ax[1, 1].set_xlim(0, 1)
ax[2, 1].set_xlim(0, 1)
ax[0,1].set_title(f'Frequency {f} Hz')
ax[1,1].set_title(f'Frequency {f+1*fs} Hz')
ax[2,1].set_title(f'Frequency {f+2*fs} Hz')

fsNyq = 2*f+1
tNyq = np.linspace(0, 1, fsNyq, endpoint=False)
sNyq = np.sin(2 * np.pi * f * tNyq)
s1Nyq = np.sin(2 * np.pi * (f + 1*fsNyq) * tNyq)
s2Nyq = np.sin(2 * np.pi * (f + 2*fsNyq) * tNyq)

ax[0, 1].stem(tNyq, sNyq)
ax[1, 1].stem(tNyq, s1Nyq)
ax[2, 1].stem(tNyq, s2Nyq)
ax[0, 1].plot(t_fin, s_fin, color='green')
ax[1, 1].plot(t_fin, s_fin, color='green')
ax[2, 1].plot(t_fin, s_fin, color='green')

plt.tight_layout()
plt.savefig("ex23.pdf")
plt.show()