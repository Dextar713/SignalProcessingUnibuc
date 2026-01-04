import os
import numpy as np
from scipy.fft import dctn
from shared_const import Q_jpeg, zigzag
from jpeg_encoder.huffman_encoder import pipeline_save

IMG_NAME = ''
COMPRESSED_DIR_NAME = ''

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


def downsample(C):
    H, W = C.shape
    return C.reshape(H//2, 2, W//2, 2).mean(axis=(1, 3))

def flatten_zigzag(block):
    N = len(block)
    flat = np.zeros(shape=N*N, dtype=int)
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

def encode_huffman(y: list[int], channel_id, H, W):
    dir_name = os.path.join(os.path.dirname(__file__), COMPRESSED_DIR_NAME)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    pipeline_save(y, H, W, output_path=os.path.join(dir_name, f'{IMG_NAME}_{channel_id}.bin'))

def encode_block(block):
    y = dctn(block)
    y_jpeg = (np.round(y / Q_jpeg).astype(int))
    y_zigzag = flatten_zigzag(y_jpeg)
    # y_rle = run_length_encode(y_zigzag)
    return y_zigzag

def encode_channel(channel, channel_id):
    N = 8
    H, W = channel.shape
    encoded_blocks = []
    for i in range(0, H-N+1, N):
        for j in range(0, W-N+1, N):
            block = channel[i:i+N, j:j+N]
            block_encoded = encode_block(block) # zigzag arr
            encoded_blocks.append(block_encoded)
    # print(max([num for block in encoded_blocks for num in block]))
    encode_huffman([num for block in encoded_blocks for num in block], channel_id, H, W)
    return encoded_blocks

def encode_jpeg(img: np.ndarray, img_name: str, compressed_dir: str):
    global IMG_NAME, COMPRESSED_DIR_NAME
    IMG_NAME = img_name
    COMPRESSED_DIR_NAME = compressed_dir
    if len(img.shape) >= 3 and img.shape[2] >= 3:
        img_ycrcb = rgb_to_ycrcb(img)
        Y, Cr, Cb = img_ycrcb[:,:,0], img_ycrcb[:,:,1], img_ycrcb[:,:,2]
        Cr = downsample(Cr)
        Cb = downsample(Cb)
        channels = [Y, Cr, Cb]
    else:
        channels = [img]
    encoded_img = []
    for i, channel in enumerate(channels):
        channel_encoded = encode_channel(channel, i)
        encoded_img.append(channel_encoded)
    return encoded_img





