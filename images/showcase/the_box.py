from mandelpy import create_image, presets, post_processing

settings = presets["the_box"]
settings.width = 8000
settings.height = 4000
settings.block_size = (1000, 1000)

img = create_image(settings, verbose=True)
img = post_processing.remove_centre_vertical_pixels(img)
img = post_processing.remove_centre_horizontal_pixels(img)
img = post_processing.blur(img)
img = img.resize((2000, 1000))

img.save("the_box.jpg", quality=90)
