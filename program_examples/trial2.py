from mandelpy import Settings, create_image, presets, power
from PIL import Image

s = presets["buddha3"]
s.fn = lambda z, c: power(z, 3)+c
create_image(s)

