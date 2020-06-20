from numba import cuda
from numba.cuda.compiler import AutoJitCUDAKernel
import typing
from cmath import *  # used for presets


class Settings:
    def __init__(self, width=1000, height=1000, left=-2, right=2, top=2, bottom=-2, max_iter=200,
                 threshold=2, tipe="mand", z0=complex(0, 0),
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

        Notes:
            Generally, high iterations are only needed if you are zooming in on the edge of the
            M-set or you are creating buddha images.

            Orbits generally require a higher threshold than 2 in order to look pretty.

            Try and generate a computationally intensive image at a small resolution first to
            see if you reall want to make it.

        Examples:
            This should create the defualt mandelbrot that everyone is used to:

            >>> from mandelpy import Settings, create_image
            >>> settings = Settings()
            >>> pillow_img = create_image(settings)
            >>> pillow_img.show()  # doctest.ignore

            To create the default buddhabrot:

            >>> from mandelpy import Settings, create_image
            >>> settings = Settings()
            >>> settings.tipe = "buddha"
            >>> pillow_img = create_image(settings)
            >>> pillow_img.show()  # doctest.ignore
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

        self.fn = fn

        self.transform = transform
        self.inv_transform = inv_transform

        self.mirror_x: bool = mirror_x
        self.mirror_y: bool = mirror_y
        self.color_scheme: int = color_scheme
        self.orbit_id: int = orbit_id

        self.block_size: typing.Tuple[2] = block_size

    @property
    def fn(self):
        return self.__fn

    @fn.setter
    def fn(self, fn):
        if fn is None:
            @cuda.jit(device=True)
            def fn(z, c):
                return z ** 2 + c

            self.__fn = fn
        elif isinstance(fn, typing.Callable):
            # turn the default function into a cuda function.
            # note that there are extreme limitations and it is
            # suggested only to use `math` and `cmath` for these functions
            self.__fn = cuda.jit(device=True)(fn)
        else:
            self.__fn = fn

    @property
    def transform(self) -> AutoJitCUDAKernel:
        return self.__transform

    @transform.setter
    def transform(self, transform: typing.Union[typing.Callable, AutoJitCUDAKernel]):
        if transform is None:
            @cuda.jit(device=True)
            def transform(z):
                return z

            self.__transform = transform
        elif isinstance(transform, typing.Callable):
            # turn the default function into a cuda function.
            # note that there are extreme limitations and it is
            # suggested only to use `math` and `cmath` for these functions
            self.__transform = cuda.jit(device=True)(transform)
        else:
            self.__transform = transform

    @property
    def inv_transform(self) -> AutoJitCUDAKernel:
        return self.__inv_transform

    @inv_transform.setter
    def inv_transform(self, inv_transform: typing.Union[typing.Callable, AutoJitCUDAKernel]):
        if inv_transform is None:
            @cuda.jit(device=True)
            def inv_transform(z):
                return z

            self.__inv_transform = inv_transform
        elif isinstance(inv_transform, typing.Callable):
            # turn the default function into a cuda function.
            # note that there are extreme limitations and it is
            # suggested only to use `math` and `cmath` for these functions
            self.__inv_transform = cuda.jit(device=True)(inv_transform)
        else:
            self.__inv_transform = inv_transform

    @property
    def frame(self) -> typing.Tuple[float, float, float, float]:
        return self.left, self.right, self.top, self.bottom

    @frame.setter
    def frame(self, frame: typing.Tuple[float, float, float, float]):
        self.left: float = frame[0]
        self.right: float = frame[1]
        self.top: float = frame[2]
        self.bottom: float = frame[3]

    @property
    def focal(self) -> typing.Tuple[float, float, float]:
        """
        Returns: A tuple of the centre real coordinate, the centre imaginary coordinate and the zoom
        """
        return (self.right + self.left) / 2, (self.top + self.bottom) / 2, (
                self.right - self.left) / 2

    @focal.setter
    def focal(self, focal: typing.Tuple[float, float, float]):
        """
        Args:
            focal: A tuple of the centre real coordinate, the centre imaginary coordinate
            and the zoom
        """
        centre_real: float = focal[0]
        centre_imag: float = focal[1]
        zoom: float = focal[2]

        if zoom < 0:  # I generally keep checks like these minimal
            raise ValueError("the third component (zoom) cannot be negative.")

        self.left: float = centre_real - zoom
        self.right: float = centre_real + zoom
        self.top: float = centre_imag + zoom
        self.bottom: float = centre_imag - zoom


@cuda.jit(device=True)
def power(z, n):
    """Finds z^n using CUDA-supported functions"""
    return exp(n * log(z))


def the_box():
    s = Settings()
    s.width = 1000
    s.height = 500
    s.max_iter = 500
    s.tipe = "buddha"
    s.frame = (-4, 4, 2, -2)
    s.transform = lambda z: tan(asin(z)) ** 2
    s.inv_transform = lambda z: sin(atan(sqrt(z)))
    s.mirror_x = True
    s.mirror_y = True
    return s


def throne():
    s = Settings()
    s.max_iter = 500
    s.tipe = "buddha"
    s.focal = (0, 0, 2.2)
    s.transform = lambda z: tan(acos(z)) ** 2
    s.inv_transform = lambda z: cos(atan(sqrt(z)))
    return s


presets = {"the_box": the_box(), "throne": throne()}
