import numpy as np
import matplotlib.pyplot as plt
import scipy.datasets

img = scipy.datasets.face()
img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
pixel_noise = 200

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=img.shape)
X_noisy = img + noise

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[1].imshow(X_noisy, cmap=plt.cm.gray)
ax[1].set_title('Noisy')

Y = np.fft.fft2(X_noisy)
Y_shifted = np.fft.fftshift(Y)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
radius = 60

y, x = np.ogrid[:rows, :cols]
mask_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
mask = np.zeros((rows, cols))
mask[mask_area] = 1

Y_filtered = Y_shifted * mask
Y_unshifted = np.fft.ifftshift(Y_filtered)
X_restored = np.fft.ifft2(Y_unshifted)
X_restored = np.real(X_restored)

ax[2].imshow(X_restored, cmap=plt.cm.gray)
ax[2].set_title(f'Restored (Low Pass, r={radius})')
plt.show()