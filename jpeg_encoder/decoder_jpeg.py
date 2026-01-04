import os
import numpy as np
from scipy.fft import idctn

from jpeg_encoder.shared_const import Q_jpeg
from shared_const import zigzag
from jpeg_encoder.huffman_encoder import pipeline_read

def ycrcb_to_rgb(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    Y  = img[:, :, 0]
    Cr = img[:, :, 1]
    Cb = img[:, :, 2]

    R = Y + 1.402   * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772   * (Cb - 128)

    out = np.stack([R, G, B], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)

def upsample(C):
    return np.repeat(np.repeat(C, 2, axis=0), 2, axis=1)

def zigzag_to_matrix(flat: np.ndarray) -> np.ndarray:
    N = 8
    M = np.zeros(shape=(N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            M[i, j] = flat[zigzag[i, j]]
    return M


def decode_block(y_zigzag: np.ndarray) -> np.ndarray:
    y_jpeg = zigzag_to_matrix(y_zigzag)
    y = y_jpeg * Q_jpeg
    x_jpeg = idctn(y)
    return x_jpeg

def decode_channel(channel_zigzag: np.ndarray, H:int, W:int) -> np.ndarray:
    decoded_channel = np.zeros(shape=(H, W), dtype=np.uint8)
    n = len(channel_zigzag)
    cur_h, cur_w = 0, 0
    for i in range(0, n-64+1, 64):
        y_zigzag = channel_zigzag[i:i + 64]
        block = decode_block(y_zigzag)
        decoded_channel[cur_h:cur_h + 8, cur_w:cur_w + 8] = block
        cur_w = (cur_w + 8) % W
        if cur_w == 0:
            cur_h = (cur_h + 8) % H
    return decoded_channel


def decode_jpeg(compressed_dir: str, img_name: str, num_channels: int) -> np.ndarray:
    channels = []
    cur_dir = os.path.dirname(__file__)
    for i in range(num_channels):
        path = os.path.join(cur_dir, compressed_dir, f'{img_name}_{i}.bin')
        vals, H, W = pipeline_read(path)
        channel = decode_channel(np.array(vals), H, W)
        channels.append(channel)
    if num_channels == 3:
        Y = channels[0]
        Cr = upsample(channels[1])
        Cb = upsample(channels[2])
        img_ycrcb = np.stack([Y, Cr, Cb], axis=-1)
        rgb_img = ycrcb_to_rgb(img_ycrcb)
        return rgb_img
    else:
        return channels[0]

