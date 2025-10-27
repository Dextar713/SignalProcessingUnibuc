import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-np.pi/2, np.pi/2, 100)
s1 = t.copy()
s2 = np.sin(t)
pade_approximation = (t - 7*t**3/60)/(1+t**2/20)

fig, ax = plt.subplots(nrows=6, ncols=1)
for i in range(len(ax)):
    ax[i].set_xlim([-np.pi/2, np.pi/2])
ax[0].plot(t, s1)
ax[1].plot(t, s2)

err_linear = np.abs(s1 - s2)
ax[2].plot(t, err_linear)
ax[3].plot(t, pade_approximation)
err_pade = np.abs(pade_approximation - s2)
ax[4].plot(t, err_pade)
ax[5].plot(t, err_pade)
ax[5].set_yscale('log')

plt.savefig('ex8.pdf')
plt.show()