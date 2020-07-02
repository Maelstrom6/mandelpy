from mandelpy import Settings, create_image

settings = Settings()  # Use the default settings to create the default mandelbrot
img = create_image(settings)  # Create a Pillow image from the settings
img.save("hello_world.png")  # Save it somewhere
