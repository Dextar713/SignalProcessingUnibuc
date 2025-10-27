import numpy as np
import matplotlib.pyplot as plt

fs = 1000
f = 200
t = [0]*3
s = [0]*3
t[0] = np.linspace(0, 1, num=fs, endpoint=False)
s[0] = np.sin(2*np.pi*f*t[0])
t[1] = t[0][::4]
s[1] = s[0][::4]
s[2] = s[0][1::4]
t[2] = t[0][1::4]

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 7))
for i in range(3):
    ax[i].plot(t[i], s[i])
    ax[i].set_xlim([0, 0.1])
plt.tight_layout()
print(t[0][0:12])
print(t[1][0:12])
plt.savefig('ex7.pdf')
plt.show()

# dupa decimare frecventa de esantionare se reduce (new_fs = fs/4)
# in decimarea b) se produce phase shift
# daca f >= fs/2 va fi efectul de aliasing (undersampling)
# in cazul acesta 1000/4 = 250 < 200*2 => aliasing
