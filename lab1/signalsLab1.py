import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 5), nrows=1, ncols=3)

time_axes = np.arange(0, 1/100, 0.0005)

s1 = lambda t: np.cos(520*np.pi*t + np.pi/3)
s2 = lambda t: np.cos(280*np.pi*t - np.pi/3)
s3 = lambda t: np.cos(120*np.pi*t + np.pi/3)

# ax[0].plot(time_axes, s1(time_axes))
# ax[1].plot(time_axes, s2(time_axes))
# ax[2].plot(time_axes, s3(time_axes))

s1_sampled = s1(time_axes)
s2_sampled = s2(time_axes)
s3_sampled = s3(time_axes)

ax[0].stem(time_axes, s1(time_axes))
ax[1].stem(time_axes, s2(time_axes))
ax[2].stem(time_axes, s3(time_axes))

#plt.savefig('xyz_sampled.pdf')

plt.show()