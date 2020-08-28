# power(z, 200) + z -1-1j

from mandelpy import create_image, Settings, power
from PIL import ImageFilter
import numpy as np
from cmath import *
from user_utilities import *
import time

images_folder = r"..\images\increasing_powers5"
video_file = r"..\test5.mp4"


def create_images():
    i = 0

    for n in np.arange(1, 5, 0.02):
        i += 1
        p = (n - 2) * abs(n - 2) + 2

        settings = Settings(transform=lambda z: power(z, p) + 1.5*z - 0.5 - 0.25j,
                            width=2000, height=2000,
                            block_size=(1000, 1000),
                            mirror_x=False)
        img = create_image(settings, verbose=True)
        img = img.filter(ImageFilter.GaussianBlur(1))
        img = img.resize((1920, 1080))
        # img.save(rf"{images_folder}\Pic{i}.jpg", optimize=True, quality=90)


if __name__ == '__main__':
    start = time.time()
    create_images()
    make_gif(images_folder, video_file, 30)
    end = time.time()
    print("Total time taken:", end - start)
