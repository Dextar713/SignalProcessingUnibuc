import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from encoder_jpeg import encode_jpeg
from jpeg_encoder.decoder_jpeg import decode_jpeg
import shared_const
from time import time

def calculate_mse(img_original, img_reconstructed):
    original = img_original.astype(np.float64)
    reconstructed = img_reconstructed.astype(np.float64)
    mse = np.mean((original - reconstructed) ** 2)
    return mse

def run_test(visual_output_flag = False, target_mse:None|float=None):
    img_name = 'stocks_usa.png'
    img_path = f'images/{img_name}'
    image = Image.open(img_path)
    # image = image.convert('L')
    W, H = image.size
    H -= H % 8
    W -= W % 8
    image = np.array(image.resize((W, H)))
    if target_mse is None:
        encode_jpeg(image, img_name, 'compressed')
        decoded = decode_jpeg('compressed', img_name, num_channels=3)
    else:
        num_iter = 10
        decoded = None
        Q_up, Q_low= 0.1, 50
        for _ in range(num_iter):
            shared_const.Q_scale = (Q_low + Q_up) / 2
            encode_jpeg(image, img_name, 'compressed')
            decoded = decode_jpeg('compressed', img_name, num_channels=3)
            cur_mse = calculate_mse(image[:, :, :3], decoded)
            print('Current mse:', cur_mse)
            if cur_mse < target_mse:
                Q_up = shared_const.Q_scale
            else:
                Q_low = shared_const.Q_scale

    if visual_output_flag:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(image)
        ax[1].imshow(decoded)
        ax[0].set_title('Original')
        ax[1].set_title('Decoded Compressed')
        plt.tight_layout()
        #plt.savefig('images/original_vs_compressed.png')
        Image.fromarray(decoded).save(f'images/compressed_{img_name}')
        # plt.show()

if __name__ == '__main__':
    start_time = time()
    run_test(visual_output_flag=True, target_mse=20**2)
    end_time = time()
    print('Total time (s):', end_time - start_time)