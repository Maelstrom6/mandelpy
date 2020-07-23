from mandelpy import create_image, Settings, power
from PIL import ImageFilter
import numpy as np
from cmath import *
from user_utilities import *
import time

images_folder = r"..\images\increasing_powers2"
video_file = r"..\test.mp4"


def create_images():
    i = 0

    for n in np.arange(1, 10, 0.02):
        i += 1
        p = (n - 2) * abs(n - 2) + 2

        settings = Settings(tipe="buddha", fn=lambda zn, c: power(zn, p) + c,
                            width=2000, height=2000,
                            block_size=(1000, 1000),
                            mirror_x=True)
        img = create_image(settings, verbose=True)
        img = img.filter(ImageFilter.GaussianBlur(1))
        img = img.resize((1920, 1080))
        img.save(rf"{images_folder}\Pic{i}.jpg", optimize=True, quality=90)


if __name__ == '__main__':
    start = time.time()
    create_images()
    make_gif(images_folder, video_file, 30)
    end = time.time()
    print("Total time taken:", end - start)
