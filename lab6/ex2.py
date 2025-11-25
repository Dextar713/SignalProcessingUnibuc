import numpy as np
import matplotlib.pyplot as plt

n = 100
t = np.linspace(0, 1, n, endpoint=False)
x = []
for _ in range(4):
    x.append(np.zeros(1))
x[0] = np.random.rand(100)
mask = (t >= 0.35) & (t <= 0.4)
x_square = []
for _ in range(4):
    x_square.append(np.zeros(n))
x_square[0][mask] = np.ones(len(t[mask]))
for i in range(1, 4):
    x[i] = x[i-1] * x[0]
    x_square[i] = x_square[i-1] * x_square[0]
plt.figure(figsize=(10,6))

for i in range(4):
    plt.subplot(4,2,2*i+1)
    plt.title('random signal')
    plt.plot(t, x[i])
    plt.subplot(4,2,2*i+2)
    plt.title('squared signal')
    plt.plot(t, x_square[i])
plt.tight_layout()
plt.savefig('ex2.pdf')
plt.show()

