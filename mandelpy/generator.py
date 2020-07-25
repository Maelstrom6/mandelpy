import numpy as np
from PIL import Image
import time
from .kernels import factories
from .settings import Settings
import typing
from .color_schemes import color


def identify_blocks(width, height, mirror_x=False, mirror_y=False, block_size=(500, 500)) -> \
        typing.List[typing.Tuple[int, int, int, int]]:
    """Identifies blocks to make from width and height. List of x, y, width, height pairs."""
    block_multiple_y = 2 if mirror_x else 1
    block_multiple_x = 2 if mirror_y else 1

    blocks = []
    for x_block in range(1 + width // (block_size[0] * block_multiple_x)):
        for y_block in range(1 + height // (block_size[1] * block_multiple_y)):
            x_offset = x_block * block_size[0]
            y_offset = y_block * block_size[1]
            block_width = min(block_size[0], width // block_multiple_x - x_offset)
            block_height = min(block_size[1], height // block_multiple_y - y_offset)
            if block_width * block_height > 0:
                blocks.append((x_offset, y_offset, block_width, block_height))

    return blocks


def compile_kernel(settings: Settings):
    """Compile the function with all available constant settings"""
    f = factories.get(settings.tipe, factories["mand"])(
        settings.width, settings.height,
        settings.left, settings.right,
        settings.top, settings.bottom,
        settings.max_iter, settings.threshold,
        settings.z0, settings.fn,
        settings.transform, settings.inv_transform,
        settings.orbit_id)

    return f


def create_image(settings: Settings, verbose: typing.Union[int, bool] = False) -> \
        typing.Union[Image.Image, None]:
    """Creates a Pillow image of a fractal using the given settings.

    Args:
        settings: The Settings object to create the image
        verbose: Whether to print its workings. If it is an `int` then gives different amounts of
            information, with amounts increasing the higher level you go. If it is `True` then
            prints at the most verbose.

    Returns:The generated Pillow Image

    Raises:
        If the dependencies have not been properly installed, it will throw some Runtime errors.

    """
    with Generator(settings, verbose) as g:
        blocks = g.blocks
        for i, block in enumerate(blocks):
            if verbose:
                print(f"Creating block {i + 1} of {len(blocks)}:", block)
            g.run_block(block)

        img = g.finished_img()
    return img


class Generator:
    def __init__(self, settings: Settings, verbose: typing.Union[int, bool] = False):
        """Initializes the object that will create the Pillow Image for you.
        After initialization, you need to get its `blocks` attribute and loop through them.
        Inside the loop you need `generator.run_block(block)` and after the loop, you need
        `img = generator.finished_img()`"""
        self.settings = settings
        self.verbose = verbose
        self.start_time = time.time()

        # The mandelbrot has a smoothing factor that results in float outputs
        if (settings.tipe == "mand") or (settings.tipe == "julia"):
            dtype = np.float
        else:
            dtype = np.int

        # Define the output array that will store pixel data
        self.output = np.zeros([settings.width, settings.height, 3], dtype=dtype)

        self.jitted_function = compile_kernel(settings)

        self.blocks = identify_blocks(settings.width, settings.height,
                                      settings.mirror_x, settings.mirror_y,
                                      settings.block_size)
        self.end_time = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def run_block(self, block):
        """Generates the numpy array of visits to particular points in the given block. This is
            done in blocks since the
            kernel does not allow jobs of size (2000, 2000)."""
        self.jitted_function[block[2], block[3]](self.output, block[0], block[1])

    def finished_img(self) -> typing.Union[Image.Image, None]:
        """Finalizes after looping to generate the final image"""
        if self.output is None:
            return

        # Finish up with mirroring operations
        if self.settings.mirror_x:
            self.output: np.ndarray = self.output + np.flip(self.output, 1)
        if self.settings.mirror_y:
            self.output: np.ndarray = self.output + np.flip(self.output, 0)

        self.end_time = time.time()
        if self.verbose > 0:
            print("Time taken:", self.end_time - self.start_time)

        output = color(self.output, self.settings.tipe,
                       self.settings.color_scheme, self.settings.max_iter)

        output = output.astype(np.uint8).transpose((1, 0, 2))
        return Image.fromarray(output)
