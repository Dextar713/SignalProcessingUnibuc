import numpy as np
import matplotlib.pyplot as plt
import time

N_arr = [128, 256, 512, 1024, 2048, 4096, 8192]
fs = 8192
f = 100
t = np.linspace(0, 1, fs, endpoint=False)
s = np.sin(2*np.pi*f*t)

my_start_time = time.time()

fig, ax = plt.subplots(nrows=1, ncols=2)
my_times = [0.0] * len(N_arr)
fft_times = [0.0] * len(N_arr)

for i, N in enumerate(N_arr):
    freq = np.zeros(N, dtype=complex)
    for k in range(N):
        sample_n = np.arange(N)
        freq[k] = np.sum(s[sample_n] * np.exp(-2j * np.pi * k * sample_n / N))
    my_end_time = time.time()

    fft_start_time = time.time()
    fft_freq = np.fft.fft(s, N)
    fft_end_time = time.time()

    my_times[i] = my_end_time - my_start_time
    fft_times[i] = fft_end_time - fft_start_time

    # print("My DFT duration:", my_end_time - my_start_time, " seconds")
    # print("FFT duration:", fft_end_time - fft_start_time, " seconds")

# ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[0].set_title("My DFT")
ax[0].plot(N_arr, my_times)
ax[1].set_title("FFT numpy")
ax[1].plot(N_arr, fft_times)
plt.tight_layout()
plt.savefig("ex1.pdf")
plt.show()


