from xmlrpc.client import DateTime
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fontTools.misc.plistlib import end_date

df = pd.read_csv("Train.csv", sep=",", index_col=0)
print(df[:3])

x = df['Count']
N = len(x)

resolution_period = (pd.to_datetime(df.iloc[1]['Datetime'], dayfirst=True) -
                     pd.to_datetime(df.iloc[0]['Datetime'], dayfirst=True))
ts = np.round(resolution_period.total_seconds())
print('Sampling period (seconds)', ts)
total_period = ((pd.to_datetime(df.iloc[-1]['Datetime'], dayfirst=True) -
                     pd.to_datetime(df.iloc[0]['Datetime'], dayfirst=True))
                + resolution_period)
print(np.round(total_period.total_seconds()/3600/24), 'days')
fs = 1 / ts
print('Sampling frequency, (Hz)', fs)

freq_bins = np.abs(np.fft.fft(x))
freq_bins = 2 * freq_bins / N
freq_bins[0] /= 2
if N % 2 == 0:
    freq_bins[N//2] /= 2
freqs = np.arange(N) * fs / N

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
ax[0].stem(freqs[:N//2], freq_bins[:N//2])
ax[0].set_xscale('log')
# plt.xlim(0, fs/2)

# x -= freq_bins[0]
freq_bins[0] = 0 # DC component of signal
top4_freqs = freqs[np.argsort(freq_bins[:N//2])[-4:]]
top4_periods = 1/top4_freqs
print("Top 4 periods (days)", top4_periods/3600/24)

df['Datetime'] = pd.to_datetime(df['Datetime'], dayfirst=True)
start_idx = 1077
for i in range(start_idx, start_idx+7):
    cur_date:datetime = df.iloc[i]['Datetime']
    if cur_date.weekday() == 0:
        start_idx = i
        break

start_date:datetime = df.iloc[start_idx]['Datetime']
days_30 = timedelta(days=30)
end_date:datetime = start_date + days_30
end_idx:int = int(start_idx + days_30.total_seconds() // ts)
print('Start montly period date', start_date)
print('End monthly period date', end_date)

ax[1].set_xlabel('Hours from start')
ax[1].set_ylabel('Count')
ax[1].set_title('Monthly period traffic')
monthly_count = x[start_idx:end_idx]
ax[1].plot(np.arange(len(monthly_count)), monthly_count)

plt.tight_layout()
plt.savefig("Traffic_Data.pdf")
