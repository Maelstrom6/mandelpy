from mandelpy import generator
from mandelpy.settings import Settings, power
from PIL import ImageFilter
import numpy as np
from cmath import *
from user_utilities import *


i = 0
for n in np.arange(1, 7, 0.02):
    i += 1

    settings = Settings(tipe="buddha", fn=lambda zn, c: power(zn, n) + c,
                        width=2000, height=2000,
                        block_size=(1000, 1000),
                        mirror_x=True)
    img = generator.create_image(settings, verbose=True)
    img = img.filter(ImageFilter.GaussianBlur(1))
    img = img.resize((1024, 1024))
    img.save(rf"C:\Users\Chris\Documents\PycharmProjects\mandelpy\images\increasing_powers\Pic"
             rf"{i}.png", optimize=True, quality=50)  # since gif is lossless, we can just use png


make_gif(r"C:\Users\Chris\Documents\PycharmProjects\mandelpy\images\increasing_powers",
         r"C:\Users\Chris\Documents\PycharmProjects\mandelpy\test.mp4", 30)
