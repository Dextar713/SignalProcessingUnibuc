import matplotlib.pyplot as plt
import numpy as np

f = 100
fs = f*2*10
T = 0.1
t = np.linspace(0,T,int(fs * T), endpoint=False)
x = np.sin(2*np.pi*f*t)
filter_dim = 200
n_filter = np.arange(filter_dim)
w_rect = np.ones(filter_dim)
w_hamming = 0.5 * (1 - np.cos(2*np.pi*n_filter/filter_dim))
#rect_output = np.convolve(x, w_rect, mode='valid')
#hamming_output = np.convolve(x, w_hamming, mode='valid')
N = len(t)
n_sliding_windows = N // filter_dim

plt.figure(figsize=(10,5), dpi=100)
plt.subplot(121)
plt.title('Rectangular filter')
for i in range(n_sliding_windows):
    start = i*filter_dim
    end = start+filter_dim
    x_axis = t[start:end]
    y_axis = x[start:end] * w_rect
    plt.plot(x_axis, y_axis, color='orange')
plt.subplot(122)
plt.title('Hamming filter')
for i in range(n_sliding_windows):
    start = i*filter_dim
    end = start+filter_dim
    x_axis = t[start:end]
    y_axis = x[start:end] * w_hamming
    plt.plot(x_axis, y_axis, color='green')
plt.tight_layout()
plt.savefig('ex5.pdf')
plt.show()