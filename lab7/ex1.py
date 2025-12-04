import matplotlib.pyplot as plt
import numpy as np
import scipy

N1, N2 = 400, 400
n1 = np.arange(N1)[:, None]
n2 = np.arange(N2)[None, :]
x = np.sin(2*np.pi*n1) + np.sin(3*np.pi*n2)
x2 = np.sin(4*np.pi*n1) + np.sin(6*np.pi*n2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
plt.imshow(x2, cmap='gray')
X = np.fft.fft2(x2)
plt.imshow(np.abs(X), cmap='gray')
plt.show()
print(x)