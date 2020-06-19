from mandelpy import generator
from mandelpy.settings import Settings
from cmath import *
import typing

settings = Settings()

# t = lambda z: tan(acos(z)) ** 2
# it = lambda z: cos(atan(sqrt(z)))
#
# settings.set_transforms(t, it)

settings.color_scheme = 3
# settings.width = 500
# settings.height = 500

img = generator.create_image(settings, verbose=True)
img.show()

