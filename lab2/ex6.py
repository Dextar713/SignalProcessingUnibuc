import numpy as np
import matplotlib.pyplot as plt

fs = 100
fv = [fs/2, fs/4, 0]
A = 1
time_axes = np.linspace(0, 1, fs, endpoint=False)
s = [0]*len(fv)

fig, ax = plt.subplots(nrows=len(fv), ncols=1)

for i in range(len(fv)):
    s[i] = np.round(A*np.sin(2*np.pi*fv[i]*time_axes), 4)
    #s[i] = time_axes
    ax[i].set_xlim(0, 1)
    ax[i].hlines(0, time_axes[0], time_axes[-1], color='r')
    ax[i].plot(time_axes, s[i])

# probabil pentru fs/2 afiseaza linia 0 din cauza
# teoremei Nyquist: fs > f*2, altfel va fi aliasing
# daca pun fs/2.01 nu mai afiseaza zero constant
plt.savefig('ex6.pdf')
plt.show()