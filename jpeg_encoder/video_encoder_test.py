import imageio.v3 as iio
import os
import cv2
from encoder_jpeg import encode_jpeg
from decoder_jpeg import decode_jpeg


def encode_video(video_path: str, compressed_dir: str):
    def read_video_frames(path):
        # 'plugin="pyav"' or "ffmpeg" allows streaming frames one by one
        # rather than loading the whole 2GB video into RAM.
        for cur_frame in iio.imread(path, plugin="pyav", index=None):
            # frame is a numpy array: (Height, Width, 3)
            yield cur_frame
    cur_idx = 0
    video_name = os.path.basename(video_path)
    print('video_name:', video_name)
    for frame in read_video_frames(video_path):
        frame_name = f'{video_name}_{cur_idx}'
        encode_jpeg(frame, frame_name, compressed_dir)
        cur_idx += 1

def decode_video(video_name: str, compressed_dir: str, output_video_path: str, fps=30):
    def frame_generator():
        cur_idx = 0
        while True:
            frame_name_id = f'{video_name}_{cur_idx}'
            expected_file_path = os.path.join(compressed_dir, f"{frame_name_id}_0.bin")

            if not os.path.exists(expected_file_path):
                print(f"Frame {cur_idx} not found. Stopping reconstruction.")
                break
            try:
                cur_frame = decode_jpeg(compressed_dir, frame_name_id, 3)
                # cur_frame = cur_frame.astype(np.uint8)
                yield cur_frame
            except Exception as e:
                print(f"Error decoding frame {cur_idx}: {e}")
                break

            cur_idx += 1

    print(f"Start decoding to {output_video_path}...")
    first_frame = next(frame_generator())
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (first_frame.shape[1], first_frame.shape[0]))
    for frame in frame_generator():
        video.write(frame)

    print("Decoded Video saved successfully.")

def run_test():
    video_dir = 'videos'
    video_name = 'highway.mp4'
    video_path = os.path.join(os.path.dirname(os.path.relpath(__file__)), video_dir, video_name)
    compressed_dir = 'video_huffman'
    output_video = 'videos/highway_restored.mp4'

    encode_video(video_path, compressed_dir)
    decode_video(video_name, compressed_dir, output_video, fps=24)

if __name__ == '__main__':
    run_test()