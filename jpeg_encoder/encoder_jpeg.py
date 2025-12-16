import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dctn, idctn

from jpeg_encoder.huffman_encoder import pipeline_save


def rgb_to_ycrcb(img):
    img = img.astype(np.float32)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    Y  =  0.299 * R + 0.587 * G + 0.114 * B
    Cr = 128 + 0.500 * R - 0.418688 * G - 0.081312 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.500 * B

    out = np.stack([Y, Cr, Cb], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


def ycrcb_to_rgb(img):
    img = img.astype(np.float32)
    Y  = img[:, :, 0]
    Cr = img[:, :, 1]
    Cb = img[:, :, 2]

    R = Y + 1.402   * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772   * (Cb - 128)

    out = np.stack([R, G, B], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


def downsample(C):
    H, W = C.shape
    return C.reshape(H//2, 2, W//2, 2).mean(axis=(1, 3))

def flatten_zigzag(block):
    N = len(block)
    zigzag = np.array([
        [1, 2, 6, 7, 15, 16, 28, 29],
        [3, 5, 8, 14, 17, 27, 30, 43],
        [4, 9, 13, 18, 26, 31, 42, 44],
        [10, 12, 19, 25, 32, 41, 45, 54],
        [11, 20, 24, 33, 40, 46, 53, 55],
        [21, 23, 34, 39, 47, 52, 56, 61],
        [22, 35, 38, 48, 51, 57, 60, 62],
        [36, 37, 49, 50, 58, 59, 63, 64]
    ]) - 1
    flat = np.zeros(shape=(N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            flat[zigzag[i, j]] = block[i, j]
    return flat

def run_length_encode(y_flat):
    y_rle = []
    n = len(y_flat)
    idx = 0
    while idx < n:
        j = idx + 1
        while j < n and y_flat[j] == y_flat[idx]:
            j += 1
        cnt = j - idx
        y_rle.append((y_flat[idx], cnt))
        idx = j
    return y_rle

def encode_huffman(y: list[int]):
    dir_name = os.path.join(os.path.dirname(__file__), 'compressed')
    num_files = len(os.listdir(dir_name))
    pipeline_save(y, output_path=os.path.join(dir_name, f'huffman_{num_files}.bin'))

def encode_block(block):
    Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 28, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]]
    y = dctn(block)
    y_jpeg = (Q_jpeg * np.round(y / Q_jpeg).astype(int))
    y_zigzag = flatten_zigzag(y_jpeg)
    # y_rle = run_length_encode(y_zigzag)
    return y_zigzag

def encode_channel(channel):
    N = 8
    H, W = channel.shape
    encoded_blocks = []
    for i in range(H-N+1):
        for j in range(W-N+1):
            block = channel[i:i+N, j:j+N]
            block_encoded = encode_block(block) # zigzag arr
            encoded_blocks.append(block_encoded)
    encode_huffman([num for block in encoded_blocks for num in block])
    return encoded_blocks

def encode_jpeg(img):
    img_ycrcb = rgb_to_ycrcb(img)
    Y, Cr, Cb = img_ycrcb
    Cr = downsample(Cr)
    Cb = downsample(Cb)
    channels = [Y, Cr, Cb]
    encoded_img = []
    for channel in channels:
        channel_encoded = encode_channel(channel)
        encoded_img.append(channel_encoded)
    return encoded_img

image = plt.imread('images/dubai.png')
encoded = encode_jpeg(image)
print(len(encoded))



