import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal
from scipy import signal

df = pd.read_csv("../lab5/Train.csv", sep=",", index_col=0)

T = 1*3600
num_days_from_start = 7
start_idx = num_days_from_start * 24
duration = 3
end_idx = start_idx + duration * 24
x = df[start_idx:end_idx]['Count']
N = len(x)

w_s = [5, 9, 13, 17]
colors = ['red', 'green', 'blue', 'orange']


#plt.figure(figsize=(10,7), dpi=100)
# for i, w in enumerate(w_s):
#     filtered = np.convolve(x, np.ones(w), mode='valid') / w
#     samples = np.arange(len(filtered))
#     plt.plot(samples, filtered, color=colors[i], label=f'w={w}')
#     plt.legend()

#plt.stem(np.arange(len(x)), x)
dc_component = np.mean(x)
x -= dc_component
X = np.abs(np.fft.fft(x))[:N//2]

#plt.stem(np.arange(len(X)), X)
strongest4_periods = np.round(T * N / np.argsort(X)[-4:] / 3600, 2)
print('Strongest 4 periods (hours)', strongest4_periods)
periods_to_filter = [2.77, 4.24, 4.5]
cutoff_period = 4.5 * 3600
cutoff_freq = 1 / cutoff_period
val_freq_norm = cutoff_freq * 2 * T
print('Normalized cutoff frequency', np.round(val_freq_norm,2))

filter_order = 5

b_butter, a_butter = signal.butter(filter_order, val_freq_norm, btype='low', output='ba')
b_cheby, a_cheby = signal.cheby1(filter_order, 5, val_freq_norm, btype='low', output='ba')
w_butter, h_butter = signal.freqz(b_butter, a_butter)
w_cheby, h_cheby = signal.freqz(b_butter, a_butter)

# plt.subplot(1, 2, 1)
# plt.semilogx(w_butter, 20 * np.log10(abs(h_butter)))
# plt.title('Butterworth filter frequency response')
# plt.xlabel('Frequency [rad/s]')
# plt.ylabel('Amplitude [dB]')
# plt.margins(0, 0.1)
# plt.grid(which='both', axis='both')
# plt.axvline(val_freq_norm, color='green') # cutoff frequency
#
# plt.subplot(1, 2, 2)
# plt.semilogx(w_cheby, 20 * np.log10(abs(h_cheby)))
# plt.title('Chebyshev filter frequency response')
# plt.xlabel('Frequency [rad/s]')
# plt.ylabel('Amplitude [dB]')
# plt.margins(0, 0.1)
# plt.grid(which='both', axis='both')
# plt.axvline(val_freq_norm, color='green') # cutoff frequency

x += dc_component
x_butter = signal.filtfilt(b_butter, a_butter, x)
x_cheby = signal.filtfilt(b_cheby, a_cheby, x)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5), sharex=True, sharey=True)
N_cheby, N_butter = x_cheby.shape[0], x_butter.shape[0]
ax[0].set_title('Raw signal')
ax[1].set_title('Butter Filtered signal')
ax[2].set_title('Chebyshev filtered signal')
ax[0].plot(np.arange(len(x)), x)
ax[1].plot(np.arange(N_butter), x_butter)
ax[2].plot(np.arange(N_cheby), x_cheby)

print('For traffic data Chebyshev >> Butterworth. Chebyshev removes more noise.')

plt.tight_layout()
#plt.savefig('Moving_average.pdf')
# plt.savefig('Traffic_FFT.pdf')
#plt.savefig('Butter_vs_Chebyshev_filter_frequency_response.pdf')
#plt.savefig('Raw_vs_Butter_vs_Chebyshev.pdf')
plt.show()
