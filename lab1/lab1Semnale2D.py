import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8), nrows=3, ncols=1)
randSignal2d = np.random.rand(128, 128)
ax[0].imshow(randSignal2d)

def foo(A):
    m, n = A.shape
    P = np.zeros((m, m))
    for i in range(m):
        P[i, m - i - 1] = 1
    return np.matmul(P, A)

randSignal2d = np.maximum(0, randSignal2d - np.linspace(0, 1, 128).reshape(-1, 1))
# print(np.eye(4) - np.eye(4) - np.array([0, 1, 2, 3]).reshape(-1, 1))
myFlippedSignal2D = foo(randSignal2d)
ax[1].imshow(randSignal2d)
ax[2].imshow(myFlippedSignal2D)

plt.savefig('FlippedSignal2D.pdf')
plt.show()

### ex3
# T = 1/f = 1/2000 = 0.0005s = 0.5ms
# 60 * 60 * 1000 / 0.5 = 7.2 * 10^6 esantioane => 7.2 * 10^6 * 4 / 8 = 3.6 * 10^6 bytes