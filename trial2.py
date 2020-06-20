from mandelpy import generator
from mandelpy.settings import Settings, presets, power
from PIL import ImageFilter
import numpy as np
from cmath import *


settings = presets["throne"]
settings.block_size = (500, 500)
settings.width = 8000
settings.height = 8000
settings.max_iter = 10000
settings.threshold = 100
settings.color_scheme = 4
img = generator.create_image(settings, verbose=True)
#img = img.filter(ImageFilter.GaussianBlur(1))
#img = img.resize((1024, 1024))
img.save("test.png")  # since gif is lossless, we can just use png



