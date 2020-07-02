from mandelpy import create_image, Settings, power
from PIL import ImageFilter
import numpy as np
from cmath import *
from user_utilities import *
import time

i = 0
start = time.time()
for n in np.arange(1, 10, 0.02):
    i += 1
    p = (n - 2) * abs(n - 2) + 2

    settings = Settings(tipe="buddha", fn=lambda zn, c: power(zn, p) + c,
                        width=2000, height=2000,
                        block_size=(1000, 1000),
                        mirror_x=True)
    img = create_image(settings, verbose=True)
    img = img.filter(ImageFilter.GaussianBlur(1))
    img = img.resize((1024, 1024))
    img.save(rf"..\images\increasing_powers2\Pic"
             rf"{i}.png", optimize=True, quality=50)  # since gif is lossless, we can just use png

make_gif(r"..\images\increasing_powers2",
         r"..\test.mp4", 30)

end = time.time()
print("Total time taken:", end - start)
