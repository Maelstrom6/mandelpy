from mandelpy import generator
from mandelpy.settings import Settings
from PIL import ImageFilter

settings = Settings()
settings.tipe = "buddha"
settings.transform = lambda z: 1/z
settings.inv_transform = lambda z: 1/z
settings.focal = (0.7, 0, 3.2)
settings.width = 8000
settings.height = 8000
settings.max_iter = 5000
settings.threshold = 2
settings.mirror_x = True
settings.color_scheme = 4

img = generator.create_image(settings, verbose=True)
img = img.filter(ImageFilter.GaussianBlur(1))
img = img.resize((1920, 1920))
img = img.crop((0, (1920-1080)/2, 1920, 1920-(1920-1080)/2))
img.save(rf"cave.png")
