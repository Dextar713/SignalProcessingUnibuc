import numpy as np
import matplotlib.pyplot as plt

n = 20
t = np.linspace(0, 1, num=n, endpoint=False)
f_sin = 3
f_cos = 8
x = np.sin(2*np.pi*f_sin*t) + np.cos(2*np.pi*f_cos*t)
d = 7
y = np.roll(x, d)
# method 1 cross correlation
cross_corr = np.fft.ifft(np.conj(np.fft.fft(x)) * np.fft.fft(y))
target_d = np.argmax(np.abs(cross_corr))
print(target_d)
assert target_d == d

# method 2 phase correlation
Y = np.fft.fft(y)
X = np.fft.fft(x)

epsilon = 1e-7
mask = np.abs(X) > epsilon

phase_corr = np.zeros_like(X)
phase_corr[mask] = Y[mask] / X[mask]
phase_diff = np.angle(Y) - np.angle(X)
phase_corr[~mask] = np.exp(1j*phase_diff[~mask])
corr = np.fft.ifft(phase_corr)
target_d_2 = np.argmax(np.abs(corr))
print(target_d_2)
assert target_d_2 == d