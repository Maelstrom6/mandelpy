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


def create_array(settings: Settings, verbose=False,
                 progress_bar=None, stop_flag=None) -> typing.Union[np.ndarray, None]:
    """Generates the numpy array of visits to particular points. This is done in blocks since the
    kernel does not allow jobs of size (2000, 2000)."""
    # The mandelbrot has a smoothing factor that results in float outputs
    if (settings.tipe == "mand") or (settings.tipe == "julia"):
        dtype = np.float
    else:
        dtype = np.int
    # Define the output array that will store pixel data
    output = np.zeros([settings.width, settings.height, 3], dtype=dtype)

    jitted_function = compile_kernel(settings)

    blocks = identify_blocks(settings.width, settings.height,
                             settings.mirror_x, settings.mirror_y,
                             settings.block_size)
    for i, block in enumerate(blocks):
        if stop_flag is not None:
            if stop_flag.value() == 1:
                return
        if progress_bar is not None:
            progress_bar.setValue(int(90*(i+1)/len(blocks)))
        if verbose:
            print(f"Creating block {i + 1} of {len(blocks)}:", block)

        jitted_function[block[2], block[3]](output, block[0], block[1])

    # Finish up with mirroring operations
    if settings.mirror_x:
        output = output + np.flip(output, 1)
    if settings.mirror_y:
        output = output + np.flip(output, 0)

    return output


def create_image(settings: Settings, verbose: typing.Union[int, bool] = False,
                 progress_bar=None, stop_flag=None) -> typing.Union[Image.Image, None]:
    """Creates a Pillow image of a fractal using the given settings.

    Args:
        settings: The Settings object to create the image
        verbose: Whether to print its workings. If it is an `int` then gives different amounts of
            information, with amounts increasing the higher level you go. If it is `True` then
            prints at the most verbose.
        progress_bar: An object that has a method `setValue(x: int)`  that will be updated in
            order to keep track of progress.
        stop_flag: An object that has a method `value()` that is a live value of whether to halt
            the execution.

    Returns:The generated Pillow Image

    Raises:
        If the dependencies have not been properly installed, it will throw some Runtime errors.

    """
    if progress_bar is not None:
        progress_bar.setValue(0)
    start_time = time.time()
    output = create_array(settings, verbose, progress_bar, stop_flag)
    end_time = time.time()

    if verbose > 0:
        print("Time taken:", end_time - start_time)

    if output is None:
        return

    output = color(output, settings.tipe, settings.color_scheme, settings.max_iter)
    if progress_bar is not None:
        progress_bar.setValue(95)

    output = output.astype(np.uint8).transpose((1, 0, 2))
    if progress_bar is not None:
        progress_bar.setValue(100)
    return Image.fromarray(output)
