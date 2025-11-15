import numpy as np
import matplotlib.pyplot as plt


def zero_pad(x: np.ndarray) -> np.ndarray:
    x = list(x)
    N = len(x)
    while N&(N-1) != 0:
        x.append(0)
        N += 1
    return np.array(x)

def my_fft_rec(x: np.ndarray) -> np.ndarray:
    N = len(x)
    if N == 0:
        return np.array([])
    if N == 1:
        return x.copy()

    unity_roots =  np.exp(-2j * np.pi * np.arange(N) / N)
    x0 = x[::2]
    x1 = x[1::2]
    y0 = my_fft_rec(x0)
    y1 = my_fft_rec(x1)
    k = N // 2
    y = np.concatenate((y0[:k] + unity_roots[:k] * y1[:k],
                        y0[:k] - unity_roots[:k] * y1[:k]))
    return y

def my_fft(x: np.ndarray) -> np.ndarray:
    N = len(x)
    if N == 0:
        return np.array([])
    if N == 1:
        return x.copy()
    if N&(N-1) != 0:
        x = zero_pad(x)
    y = my_fft_rec(x)
    return y


def my_ifft_rec(y: np.ndarray) -> np.ndarray:
    N = len(y)
    if N == 0:
        return np.array([])
    if N == 1:
        return y.copy()

    unity_roots =  np.exp(2j * np.pi * np.arange(N) / N)
    y0 = y[::2]
    y1 = y[1::2]
    x0 = my_ifft_rec(y0)
    x1 = my_ifft_rec(y1)
    k = N // 2
    x = np.concatenate((x0[:k] + unity_roots[:k] * x1[:k],
                        x0[:k] - unity_roots[:k] * x1[:k]))
    return x

def my_ifft(y: np.ndarray) -> np.ndarray:
    N = len(y)
    if N == 0:
        return np.array([])
    if N == 1:
        return y.copy()
    if N&(N-1) != 0:
        y = zero_pad(y)
    y = my_ifft_rec(y)
    return y / N


def test_fft():
    fs = 16
    N = fs
    f1, f2 = 2, 7
    t = np.linspace(0, 1, fs, endpoint=False)
    x = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)
    X = my_fft(x)
    X_numpy = np.fft.fft(x)
    if N&(N-1) == 0:
        assert np.allclose(np.abs(X_numpy), np.abs(X))
    f_analysis = np.arange(len(X)) * fs / len(X)
    plt.stem(f_analysis[:len(X)//2], np.abs(X)[:len(X)//2])
    plt.savefig("My_FFT.pdf")

def test_ifft():
    fs = 16
    f1, f2 = 2, 7
    t = np.linspace(0, 1, fs, endpoint=False)
    x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    X = my_fft(x)
    X_numpy = np.fft.fft(x)
    N = len(x)
    if N&(N-1) == 0:
        assert np.allclose(np.abs(X_numpy), np.abs(X))
    x_ifft = my_ifft(X)
    x_numpy_ifft = np.fft.ifft(X)
    if N&(N-1) == 0:
        assert np.allclose(np.abs(x_ifft), np.abs(x_numpy_ifft))
        assert np.allclose(np.abs(x_ifft), np.abs(x))


if __name__ == "__main__":
    test_fft()
    test_ifft()