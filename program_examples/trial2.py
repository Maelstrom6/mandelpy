from mandelpy import Settings, create_image

s = Settings()
img = create_image(s)
img = img.thumbnail((800, 800))
print(img)

