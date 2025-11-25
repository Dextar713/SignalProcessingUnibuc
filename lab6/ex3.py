import sys

import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../custom_fft')
from custom_fft import my_fft, my_ifft
from time import time

n = 10000
p = np.random.randint(1, 11, size=n+1)
q = np.random.randint(1, 11, size=n+1)

def convolution(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    n1, n2 = len(p), len(q)
    r = np.zeros(n1+n2-1)
    for i in range(n1):
        for j in range(n2):
            r[i+j] += p[i] * q[j]
    return r

def multiply_with_fft(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    n1, n2 = len(p), len(q)
    p_padded = np.append(p, [0]*(2*n+1-n1))
    q_padded = np.append(q, [0] * (2 * n + 1 - n2))
    p_fft = my_fft(p_padded)
    q_fft = my_fft(q_padded)
    r_fft = p_fft * q_fft
    r = my_ifft(r_fft)
    return np.real(r[:2*n+1])

def multiply_with_numpy_fft(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    n1, n2 = len(p), len(q)
    p_padded = np.append(p, [0]*(2*n+1-n1))
    q_padded = np.append(q, [0] * (2 * n + 1 - n2))
    p_fft = my_fft(p_padded)
    q_fft = my_fft(q_padded)
    r_fft = p_fft * q_fft
    r = my_ifft(r_fft)
    return np.real(r[:2*n+1])

start = time()
res = convolution(p, q)
end = time()
convolution_duration = end - start
start = time()
res_fft = multiply_with_fft(p, q)
end = time()
fft_duration = end - start
start = time()
res_numpy_fft = multiply_with_numpy_fft(p, q)
end = time()
numpy_fft_duration = end - start
#print('Original p', p)
#print('Original q', q)
#print('Multiplication result: ', res)
assert np.allclose(res, res_fft)
assert np.allclose(res_fft, res_numpy_fft)

print('Convolution time: ', np.round(convolution_duration, 2), 's')
print('My FFT time: ', np.round(fft_duration,2), 's')
print('Numpy FFT time: ', np.round(numpy_fft_duration,2), 's')

