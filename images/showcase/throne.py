from mandelpy import generator
from mandelpy.settings import Settings, presets
from PIL import ImageFilter, Image

settings = presets["throne"]
w = 8000
h = 8000
settings.width = w
settings.height = h
settings.max_iter = 10000
settings.color_scheme = 4
settings.threshold = 100

img = generator.create_image(settings, verbose=True)

# remove the middle cross of pixels
top_left = img.crop((0, 0, h/2 - 1, w/2 -1))
top_right = img.crop((0, w/2+1, h/2-1, w))
bottom_left = img.crop((h/2+1, 0, h, w/2-1))
bottom_right = img.crop((h/2+1, w/2+1, h, w))
img = Image.new("RGB", (w-2, h-2))
img.paste(top_left, (0, 0))
img.paste(top_right, (0, w//2-1))
img.paste(bottom_left, (h//2-1, 0))
img.paste(bottom_right, (w//2-1, h//2-1))
img = img.filter(ImageFilter.GaussianBlur(1))
img = img.resize((2000, 2000))
img = img.rotate(90)
img.save(rf"throne.png")
