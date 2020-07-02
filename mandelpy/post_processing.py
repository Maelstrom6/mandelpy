from PIL import Image, ImageEnhance, ImageFilter


def remove_centre_horizontal_pixels(img: Image.Image) -> Image.Image:
    """Removes the middle pixels of the image that form a horizontal line of height 2"""
    width, height = img.size
    top = img.crop((0, 0, width, height // 2 - 1))
    bottom = img.crop((0, height // 2 + 1, width, height))
    img = Image.new("RGB", (width, height - 2))
    img.paste(top, (0, 0))
    img.paste(bottom, (0, height // 2 - 1))
    return img


def remove_centre_vertical_pixels(img: Image.Image) -> Image.Image:
    """Removes the middle pixels of the image that form a vertical line of width 2"""
    width, height = img.size
    left = img.crop((0, 0, width // 2 - 1, height))
    right = img.crop((width // 2 + 1, 0, width, height))
    img = Image.new("RGB", (width - 2, height))
    img.paste(left, (0, 0))
    img.paste(right, (width // 2 - 1, 0))
    return img


def enhance(img: Image.Image, h: float = 1, s: float = 1, b: float = 1) -> Image.Image:
    """Enhances the hue, saturation and brightness of an image with given factors.
    I know that color != hue and contrast != saturation but they're close enough for me."""
    en = ImageEnhance.Color(img)
    img = en.enhance(h)

    en = ImageEnhance.Contrast(img)
    img = en.enhance(s)

    en = ImageEnhance.Brightness(img)
    img = en.enhance(b)

    return img


def blur(img: Image.Image, radius: int = 1) -> Image.Image:
    """Performs a Gaussian blur"""
    return img.filter(ImageFilter.GaussianBlur(radius))
