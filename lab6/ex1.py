import numpy as np
import matplotlib.pyplot as plt

B = 1
T_edges = [-3, 3]
fs = [1, 1.5, 2, 4]
times = []
for f in fs:
    t = np.linspace(start=T_edges[0], stop=T_edges[1],
                    num=int(f*(T_edges[1] - T_edges[0])), endpoint=False)
    times.append(t)
x = []
for t in times:
    vals = np.sinc(B * t)**2
    x.append(vals)

x_hat = []
fs_fin = 100
t_fine = np.linspace(T_edges[0], T_edges[1], fs_fin,  endpoint=False)
for i in range(len(x)):
    res = np.zeros_like(t_fine)
    for n in range(len(times[i])):
        res += x[i][n] * np.sinc((t_fine - times[i][n])*fs[i])
    x_hat.append(res)

plt.figure(figsize=(10, 7))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.stem(times[i], x[i])
    plt.title(f'Frequency of sinc (Hz): {fs[i]}')
    plt.plot(t_fine, x_hat[i], color='orange')
plt.tight_layout()
plt.savefig('ex1.pdf')
plt.show()