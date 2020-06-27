from mandelpy import generator
from mandelpy.settings import Settings, presets, power
from PIL import ImageFilter
import numpy as np
from numba import cuda
from cmath import *

@cuda.jit(device=True)
def inv(z):
    return 1 / z

@cuda.jit(device=True)
def square(z):
    return z ** 2


functions = [inv, exp, sin, cos, tan, square, log,
             asin, acos, atan, sqrt, sinh, cosh, tanh, asinh, acosh, atanh]
inv_functions = [inv, log, asin, acos, atan, sqrt,
                 exp, sin, cos, tan, square, asinh, acosh, atanh, sinh, cosh, tanh]

# 9 14
# 16 5

# for i, f in enumerate(functions):
#     f_inv = inv_functions[i]
#     settings = Settings()
#     settings.tipe = "buddha"
#     settings.transform = lambda z: f(z)
#     settings.inv_transform = lambda z: f_inv(z)
#     settings.focal = (0, 0, 4)
#     settings.block_size = (1000, 1000)
#     settings.color_scheme = 4
#     img = generator.create_image(settings, verbose=True)
#     img.save(rf"C:\Users\Chris\Documents\PycharmProjects\mandelpy\images\buddha1\buddha"
#              rf"{i:03}.png")

for i, f1 in enumerate(functions):
    for j, f2 in enumerate(functions):
        if (i < 11) and (j < 11):
            continue
        f_inv1 = inv_functions[i]
        f_inv2 = inv_functions[j]
        settings = Settings()
        settings.tipe = "buddha"
        settings.transform = lambda z: f2(f1(z))
        settings.inv_transform = lambda z: f_inv1(f_inv2(z))
        settings.focal = (0, 0, 4)
        settings.block_size = (1000, 1000)
        settings.color_scheme = 4
        img = generator.create_image(settings, verbose=True)
        img.save(rf"C:\Users\Chris\Documents\PycharmProjects\mandelpy\images\buddha2\buddha"
                 rf"{i:03} {j:03}.png")
