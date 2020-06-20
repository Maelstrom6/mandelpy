from PIL import Image, ImageFilter

img = Image.open('test.png')
top_left = img.crop((0, 0, 3999, 3999))
top_right = img.crop((0, 4001, 3999, 8000))
bottom_left = img.crop((4001, 0, 8000, 3999))
bottom_right = img.crop((4001, 4001, 8000, 8000))
img = Image.new('RGB', (7998, 7998))
img.paste(top_left, (0, 0))
img.paste(top_right, (0, 3999))
img.paste(bottom_left, (3999, 0))
img.paste(bottom_right, (3999, 3999))
new_img = img.resize((2000, 2000), resample=Image.BICUBIC)
new_img.save("test1.png", "PNG")

