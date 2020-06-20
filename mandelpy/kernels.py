from numba import cuda
import numba
from cmath import *
import math


def buddha_factory(width: int, height: int, left: float, right: float, top: float, bottom: float,
                   max_iter: int, threshold: float, z0: complex, fn, transform, inv_transform,
                   orbit_id):
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
        orbit_id: not used

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

        id_x = cuda.blockIdx.x + offset_x
        id_y = cuda.threadIdx.x + offset_y

        c = pixel_to_point(id_x, id_y)
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
                        transform, inv_transform, orbit_id):
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
        orbit_id: not used

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

        id_x = cuda.blockIdx.x + offset_x
        id_y = cuda.threadIdx.x + offset_y

        c = pixel_to_point(id_x, id_y)
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
                       transform, inv_transform, orbit_id):
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
        orbit_id: not used

    Returns:
        The compiled, jitted function for the buddhabrot

    """

    @cuda.jit
    def mandelbrot(data, offset_x, offset_y):
        def pixel_to_point(x, y):
            return complex(left + (right - left) * x / width,
                           top + (bottom - top) * y / height)

        id_x = cuda.blockIdx.x + offset_x
        id_y = cuda.threadIdx.x + offset_y

        c = pixel_to_point(id_x, id_y)
        c = transform(c)

        zn = z0
        zn = fn(zn, c)
        # can't dynamically define array size
        escaped = False

        for i in range(max_iter):
            zn = fn(zn, c)
            if zn.real ** 2 + zn.imag ** 2 > threshold ** 2:
                escaped = True
                # the smoothing factor
                nu = math.log(math.log(abs(zn)) / math.log(2.0)) / math.log(2.0)
                data[id_x, id_y] = i - nu
                break

        if not escaped:
            data[id_x, id_y] = max_iter

    return mandelbrot


def julia_factory(width: int, height: int, left: float, right: float, top: float,
                  bottom: float, max_iter: int, threshold: float, z0: complex, fn,
                  transform, inv_transform, orbit_id):
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
        orbit_id: not used

    Returns:
        The compiled, jitted function for the buddhabrot

    """

    @cuda.jit
    def julia(data, offset_x, offset_y):
        def pixel_to_point(x, y):
            return complex(left + (right - left) * x / width,
                           top + (bottom - top) * y / height)

        id_x = cuda.blockIdx.x + offset_x
        id_y = cuda.threadIdx.x + offset_y

        c = z0
        # c = transform(c)

        zn = pixel_to_point(id_x, id_y)
        zn = transform(zn)
        # can't dynamically define array size
        escaped = False

        for i in range(max_iter):
            zn_min_one = zn
            zn = fn(zn, c)
            if zn.real ** 2 + zn.imag ** 2 > threshold ** 2:
                escaped = True
                # the smoothing factor
                nu = 1 - (threshold - abs(zn)) / (abs(zn) - abs(zn_min_one))
                data[id_x, id_y] = i - nu
                break

        if not escaped:
            data[id_x, id_y] = max_iter

    return julia


def julai_buddha_factory(width: int, height: int, left: float, right: float, top: float,
                         bottom: float,
                         max_iter: int, threshold: float, z0: complex, fn, transform,
                         inv_transform, orbit_id):
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
        orbit_id: not used

    Returns:
        The compiled, jitted function for the buddhabrot

    """

    @cuda.jit
    def julia_buddha(data, offset_x, offset_y):
        def pixel_to_point(x, y):
            return complex(left + (right - left) * x / width,
                           top + (bottom - top) * y / height)

        def point_to_pixel(x_point, y_point):
            return round((x_point - left) * width / (right - left)), \
                   round((y_point - top) * height / (bottom - top))

        id_x = cuda.blockIdx.x + offset_x
        id_y = cuda.threadIdx.x + offset_y

        zn = pixel_to_point(id_x, id_y)
        zn = transform(zn)

        c = z0
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

    return julia_buddha


def orbit_factory(width: int, height: int, left: float, right: float, top: float,
                  bottom: float, max_iter: int, threshold: float, z0: complex, fn,
                  transform, inv_transform, orbit_id):
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
        orbit_id: The type of orbit to do

    Returns:
        The compiled, jitted function for the buddhabrot

    """

    @cuda.jit
    def orbits(data, offset_x, offset_y):
        def pixel_to_point(x, y):
            return complex(left + (right - left) * x / width,
                           top + (bottom - top) * y / height)

        id_x = cuda.blockIdx.x + offset_x
        id_y = cuda.threadIdx.x + offset_y

        c = pixel_to_point(id_x, id_y)
        c = transform(c)

        zn = complex(0, 0)
        trap = z0
        distance = 100000

        if orbit_id == 1:  # crosses
            for i in range(max_iter):
                zn = fn(zn, c)
                if zn.real ** 2 + zn.imag ** 2 > threshold ** 2:
                    data[id_x, id_y] = i
                    break
                hor_dist = abs(zn.real - trap.real)
                ver_dist = abs(zn.imag - trap.imag)
                if distance > hor_dist:
                    distance = hor_dist
                if distance > ver_dist:
                    distance = ver_dist

        elif orbit_id == 2:  # ring
            max_acceptable_dist = 0.5
            min_acceptable_dist = 0.4
            for i in range(max_iter):
                zn = fn(zn, c)
                if zn.real ** 2 + zn.imag ** 2 > threshold ** 2:
                    data[id_x, id_y] = i
                    break
                point_dist = abs(zn - trap)
                if (distance > point_dist) \
                        and (min_acceptable_dist < point_dist < max_acceptable_dist):
                    distance = point_dist

        else:  # dot
            for i in range(max_iter):
                zn = fn(zn, c)
                if zn.real ** 2 + zn.imag ** 2 > threshold ** 2:
                    data[id_x, id_y] = i
                    break
                point_dist = abs(zn - trap)
                if distance > point_dist:
                    distance = point_dist

        data[id_x, id_y] = 100 * distance

    return orbits


factories = {
    "buddha": buddha_factory,
    "mand": mandelbrot_factory,
    "julia": julia_factory,
    "julia_buddha": julai_buddha_factory,
    "orbit": orbit_factory}
