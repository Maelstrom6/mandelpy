from numba import cuda
import numba
from cmath import *
import math


def buddha_factory(width: int, height: int, left: float, right: float, top: float, bottom: float,
                   max_iter: int, threshold: float, z0: complex, fn, transform, inv_transform):
    """A wrapper function for the buddha kernel so that it compiles after inputting the above args

    Args:
        width: The entire width in pixels of the image
        height: The entire height in pixels of the image
        left: The minimum value of the real component of the plane
        right: The maximum value of the real component of the plane
        top: The maximum value of the imaginary component of the plane
        bottom: The minimum value of the imaginary component of the plane
        max_iter: The maximum number of iterations allowed
        threshold: The minimum magnitude of an iteration to reach before being considered escaped
        z0: The starting point for the iterations
        fn: The function to be iterated
        transform: The function that transforms input points
        inv_transform: The inverse function of `transform`

    Returns:
        The compiled, jitted function for the buddhabrot

    """

    @cuda.jit
    def buddha(data, offset_x, offset_y):
        def pixel_to_point(x, y):
            return complex(left + (right - left) * x / width,
                           top + (bottom - top) * y / height)

        def point_to_pixel(x_point, y_point):
            return round((x_point - left) * width / (right - left)), \
                   round((y_point - top) * height / (bottom - top))

        id_x = cuda.threadIdx.x
        id_y = cuda.blockIdx.x

        c = pixel_to_point(id_x + offset_x, id_y + offset_y)
        c = transform(c)

        zn = z0
        zn = fn(zn, c)
        visited_coords = cuda.local.array(max_iter, dtype=numba.complex64)
        # can't dynamically define array size

        for i in range(max_iter):
            zn = fn(zn, c)
            visited_coords[i] = zn

            if zn.real ** 2 + zn.imag ** 2 > threshold ** 2:
                j = 0
                while visited_coords[j] != 0:
                    # for j, coord in enumerate(visited_coords):
                    coord = inv_transform(visited_coords[j])
                    x_pixel, y_pixel = point_to_pixel(coord.real, coord.imag)
                    if (0 < y_pixel < height) and (0 < x_pixel < width):
                        if j < 0.01 * max_iter:
                            data[x_pixel, y_pixel, 2] += 1
                        if j < 0.1 * max_iter:
                            data[x_pixel, y_pixel, 1] += 1
                        data[x_pixel, y_pixel, 0] += 1
                    j += 1
                break

    return buddha


def anti_buddha_factory(width: int, height: int, left: float, right: float, top: float,
                        bottom: float, max_iter: int, threshold: float, z0: complex, fn,
                        transform, inv_transform):
    """A wrapper function for the buddha kernel so that it compiles after inputting the above args

    Args:
        width: The entire width in pixels of the image
        height: The entire height in pixels of the image
        left: The minimum value of the real component of the plane
        right: The maximum value of the real component of the plane
        top: The maximum value of the imaginary component of the plane
        bottom: The minimum value of the imaginary component of the plane
        max_iter: The maximum number of iterations allowed
        threshold: The minimum magnitude of an iteration to reach before being considered escaped
        z0: The starting point for the iterations
        fn: The function to be iterated
        transform: The function that transforms input points
        inv_transform: The inverse function of `transform`

    Returns:
        The compiled, jitted function for the buddhabrot

    """

    @cuda.jit
    def anti_buddha(data, offset_x, offset_y):
        def pixel_to_point(x, y):
            return complex(left + (right - left) * x / width,
                           top + (bottom - top) * y / height)

        def point_to_pixel(x_point, y_point):
            return round((x_point - left) * width / (right - left)), \
                   round((y_point - top) * height / (bottom - top))

        id_x = cuda.threadIdx.x
        id_y = cuda.blockIdx.x

        c = pixel_to_point(id_x + offset_x, id_y + offset_y)
        c = transform(c)

        zn = z0
        zn = fn(zn, c)
        visited_coords = cuda.local.array(max_iter, dtype=numba.complex64)
        # can't dynamically define array size
        escaped = False

        for i in range(max_iter):
            zn = fn(zn, c)
            visited_coords[i] = zn
            if zn.real ** 2 + zn.imag ** 2 > threshold ** 2:
                escaped = True
                break

        if not escaped:
            j = 0
            while visited_coords[j] != 0:
                # for j, coord in enumerate(visited_coords):
                coord = inv_transform(visited_coords[j])
                x_pixel, y_pixel = point_to_pixel(coord.real, coord.imag)
                if (0 < y_pixel < height) and (0 < x_pixel < width):
                    if j < 0.01 * max_iter:
                        data[x_pixel, y_pixel, 2] += 1
                    if j < 0.1 * max_iter:
                        data[x_pixel, y_pixel, 1] += 1
                    data[x_pixel, y_pixel, 0] += 1
                j += 1

    return anti_buddha


def mandelbrot_factory(width: int, height: int, left: float, right: float, top: float,
                       bottom: float, max_iter: int, threshold: float, z0: complex, fn,
                       transform):
    """A wrapper function for the buddha kernel so that it compiles after inputting the above args

    Args:
        width: The entire width in pixels of the image
        height: The entire height in pixels of the image
        left: The minimum value of the real component of the plane
        right: The maximum value of the real component of the plane
        top: The maximum value of the imaginary component of the plane
        bottom: The minimum value of the imaginary component of the plane
        max_iter: The maximum number of iterations allowed
        threshold: The minimum magnitude of an iteration to reach before being considered escaped
        z0: The starting point for the iterations
        fn: The function to be iterated
        transform: The function that transforms input points

    Returns:
        The compiled, jitted function for the buddhabrot

    """

    @cuda.jit
    def mandelbrot(data, offset_x, offset_y):
        def pixel_to_point(x, y):
            return complex(left + (right - left) * x / width,
                           top + (bottom - top) * y / height)

        id_x = cuda.threadIdx.x
        id_y = cuda.blockIdx.x

        c = pixel_to_point(id_x + offset_x, id_y + offset_y)
        c = transform(c)

        zn = z0
        zn = fn(zn, c)
        visited_coords = cuda.local.array(max_iter, dtype=numba.complex64)
        # can't dynamically define array size
        escaped = False

        for i in range(max_iter):
            zn = fn(zn, c)
            visited_coords[i] = zn
            if zn.real ** 2 + zn.imag ** 2 > threshold ** 2:
                escaped = True
                # the smoothing factor
                nu = math.log(math.log(abs(zn)) / math.log(2.0)) / math.log(2.0)
                data[id_x + offset_x, id_y + offset_y] = i - nu
                break

        if not escaped:
            data[id_x + offset_x, id_y + offset_y] = max_iter

    return mandelbrot
