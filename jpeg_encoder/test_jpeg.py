import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from encoder_jpeg import encode_jpeg
from jpeg_encoder.decoder_jpeg import decode_jpeg
from time import time

def run_test():
    img_name = 'stocks_usa.png'
    img_path = f'images/{img_name}'
    image = Image.open(img_path)
    # image = image.convert('L')
    W, H = image.size
    H -= H % 8
    W -= W % 8
    image = np.array(image.resize((W, H)))
    encode_jpeg(image, img_name, 'compressed')
    decoded = decode_jpeg('compressed', img_name, num_channels=3)
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
    run_test()
    end_time = time()
    print('Total time (s):', end_time - start_time)