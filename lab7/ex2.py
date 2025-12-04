import numpy as np
import matplotlib.pyplot as plt
import scipy.datasets

img = scipy.datasets.face()
img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
#plt.imshow(img, cmap='gray')

Y = np.fft.fft2(img)
freq_db = 20*np.log10(abs(Y))
#freq_db = np.fft.fftshift(freq_db)
#plt.imshow(freq_db)
H, W = Y.shape
center = (W//2, H//2)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
freq_cutoff = 130
Y_cutoff = Y.copy()
Y_cutoff[freq_db > freq_cutoff] = 0
X_cutoff = np.fft.ifft2(Y_cutoff)
X_cutoff = np.real(X_cutoff)


ax[0].imshow(img, cmap='gray')
ax[1].imshow(X_cutoff, cmap='gray')
plt.show()