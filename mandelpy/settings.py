from numba import cuda
from numba.cuda.compiler import AutoJitCUDAKernel
import typing


class Settings:
    def __init__(self, width=1000, height=1000, left=-2, right=2, top=2, bottom=-2, max_iter=200,
                 threshold=2, tipe="buddha", z0=complex(0, 0),
                 fn: typing.Union[typing.Callable, AutoJitCUDAKernel] = None,
                 transform: typing.Union[typing.Callable, AutoJitCUDAKernel] = None,
                 inv_transform: typing.Union[typing.Callable, AutoJitCUDAKernel] = None,
                 mirror_x: bool = False, mirror_y: bool = False,
                 color_scheme=1, orbit_id=1, block_size=(500, 500)):
        """The main settings object to create mandelbrot images

        Args:
            width: The entire width in pixels of the image
            height: The entire height in pixels of the image
            left: The minimum value of the real component of the plane
            right: The maximum value of the real component of the plane
            top: The maximum value of the imaginary component of the plane
            bottom: The minimum value of the imaginary component of the plane
            max_iter: The maximum number of iterations allowed
            threshold: The minimum magnitude of an iteration to reach before being considered
                escaped
            tipe: The type of mandelbrot to make, such as "buddha", "mand", "orbit",
                "antibuddha", "julia". The name was used to not override Python's `type`.
            z0: The starting value z for tipes such as "orbit" or "julia"
            fn: The function to be iterated
            transform: The function that transforms input points
            inv_transform: The inverse function of `transform`
            mirror_x: Whether to mirror across the x-axis (if True, will roughly double the speed)
            mirror_y: Whether to mirror across the y-axis (if True, will roughly double the speed)
            color_scheme: The ID of the color scheme
            orbit_id: The ID of the orbit type if tipe is "orbit"
            block_size: CUDA cannot generate a 10000 by 10000 image with one job. It needs to be
                separated into blocks. Each block specifies the pixels/complex points to find.
                `block_size` is the maximum size of any given block. It is not advisable to go
                larger than (1000, 1000).

        Warnings:
            If `width` or `height` is not an integer multiple of `block_size` and `mirror_x`
            and/or `mirror_y` is True, visual artifacts may appear. I am fairly, confident that
            there are no bugs and it might be something on CUDA's side.
            If you do find that there is a bug causing this, please put in a PR.

        Warnings:
            If you want to change `fn` or `transform` or inv_transform` after this initialization,
            use set_fn or other applicable setter methods.
        """

        self.width: int = width
        self.height: int = height
        self.left: float = left
        self.right: float = right
        self.top: float = top
        self.bottom: float = bottom
        self.max_iter: int = max_iter
        self.threshold: float = threshold
        self.tipe: str = tipe
        self.z0: complex = z0

        self.fn = None
        self.set_fn(fn)

        self.transform = None
        self.inv_transform = None
        self.set_transforms(transform, inv_transform)

        self.mirror_x: bool = mirror_x
        self.mirror_y: bool = mirror_y
        self.color_scheme: int = color_scheme
        self.orbit_id: int = orbit_id

        self.block_size: typing.Tuple[2] = block_size

    def set_fn(self, fn):
        if fn is None:
            @cuda.jit(device=True)
            def fn(z, c):
                return z ** 2 + c

            self.fn = fn
        elif isinstance(fn, typing.Callable):
            # turn the default function into a cuda function.
            # note that there are extreme limitations and it is
            # suggested only to use `math` and `cmath` for these functions
            self.fn = cuda.jit(device=True)(fn)
        else:
            self.fn = fn

    def set_transforms(self, transform, inv_transform):
        if (transform is None) or (inv_transform is None):
            @cuda.jit(device=True)
            def transform(z):
                return z

            @cuda.jit(device=True)
            def inv_transform(z):
                return z

            self.transform = transform
            self.inv_transform = inv_transform
        elif isinstance(transform, typing.Callable) and isinstance(inv_transform, typing.Callable):
            # turn the default function into a cuda function.
            # note that there are extreme limitations and it is
            # suggested only to use `math` and `cmath` for these functions
            self.transform = cuda.jit(device=True)(transform)
            self.inv_transform = cuda.jit(device=True)(inv_transform)
        else:
            self.transform = transform
            self.inv_transform = inv_transform

    def set_frame(self, left: float, right: float, top: float, bottom: float):
        self.left: float = left
        self.right: float = right
        self.top: float = top
        self.bottom: float = bottom

    def set_focal(self, centre_real: float, centre_imag: float, zoom: float):
        self.left: float = centre_real - zoom
        self.right: float = centre_real + zoom
        self.top: float = centre_imag + zoom
        self.bottom: float = centre_imag - zoom

