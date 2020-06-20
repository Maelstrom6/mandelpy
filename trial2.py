from mandelpy import generator
from mandelpy.settings import Settings, presets
from cmath import *

settings = Settings()

# t = lambda z: tan(acos(z)) ** 2
# it = lambda z: cos(atan(sqrt(z)))
#
# settings.set_transforms(t, it)
# settings.transform = lambda z: 1/z
# settings.transform = lambda z: 1/z
# # settings.set_transforms(t, it)
# settings.tipe = "buddha"
# settings.orbit_id = 1
# settings.threshold = 2
# settings.color_scheme = 4
# settings.focal = (1, 0, 3)
# settings.width = 500
# settings.height = 500
settings = presets["the_box"]
img = generator.create_image(settings, verbose=True)
img.show()


