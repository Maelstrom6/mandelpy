from numba import cuda
import numba
from cmath import *
import math


def pixel_to_point_wrapper(width, height, left, right, top, bottom):
    @cuda.jit(device=True)
    def pixel_to_point(x, y):
        return complex(left + (right - left) * x / width,
                       top + (bottom - top) * y / height)

    return pixel_to_point


def point_to_pixel_wrapper(width, height, left, right, top, bottom):
    @cuda.jit(device=True)
    def point_to_pixel(x_point, y_point):
        return round((x_point - left) * width / (right - left)), \
               round((y_point - top) * height / (bottom - top))

    return point_to_pixel


@cuda.jit(device=True)
def is_in_main_bulbs(c):
    # if in the main bulb
    w = 0.25 - c
    if abs(w) < (math.cos(abs(phase(w)) / 2)) ** 2:
        return True

    # if in secondary bulb
    dist_from_min_one = abs(c + 1)
    if dist_from_min_one < 0.25:
        return True
    return False


@cuda.jit(device=True)
def identify_ids(offset_x, offset_y):
    id_x = cuda.blockIdx.x + offset_x
    id_y = cuda.threadIdx.x + offset_y
    return id_x, id_y


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
    pixel_to_point = pixel_to_point_wrapper(width, height, left, right, top, bottom)
    point_to_pixel = point_to_pixel_wrapper(width, height, left, right, top, bottom)

    @cuda.jit
    def buddha(data, offset_x, offset_y):
        id_x, id_y = identify_ids(offset_x, offset_y)
        c = transform(pixel_to_point(id_x, id_y))
        zn = z0

        # cycle detection algorithm initial values
        check_step = 1
        epsilon = (right - left) / 1000000000  # scales as you zoom in
        zn_cycle = c

        if is_in_main_bulbs(c):
            return

        zn = fn(zn, c)  # iterate once so we don't add to visited_coords for no reason
        visited_coords = cuda.local.array(max_iter, dtype=numba.complex64)

        for i in range(max_iter):
            zn = fn(zn, c)
            visited_coords[i] = zn

            # the finite iteration algorithm
            if zn.real ** 2 + zn.imag ** 2 > threshold ** 2:
                j = 0
                while visited_coords[j] != 0:
                    coord = inv_transform(visited_coords[j])
                    x_pixel, y_pixel = point_to_pixel(coord.real, coord.imag)
                    if (0 < y_pixel < height) and (0 < x_pixel < width):
                        if (j < 10):
                            data[x_pixel, y_pixel, 2] += 1
                        if j < 100:
                            data[x_pixel, y_pixel, 1] += 1
                        data[x_pixel, y_pixel, 0] += 1
                    j += 1
                break

            # cycle detection algorithm
            if i > check_step:
                if abs(zn - zn_cycle) < epsilon:
                    return
                if i == check_step * 2:
                    check_step *= 2
                    zn_cycle = zn

    return buddha


def anti_buddha_factory(width: int, height: int, left: float, right: float, top: float,
                        bottom: float, max_iter: int, threshold: float, z0: complex, fn,
                        transform, inv_transform, orbit_id):
    """A wrapper function for the anti-buddha kernel so that it compiles after inputting the above
    args

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

    pixel_to_point = pixel_to_point_wrapper(width, height, left, right, top, bottom)
    point_to_pixel = point_to_pixel_wrapper(width, height, left, right, top, bottom)

    @cuda.jit
    def anti_buddha(data, offset_x, offset_y):
        id_x, id_y = identify_ids(offset_x, offset_y)
        c = transform(pixel_to_point(id_x, id_y))
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
    """A wrapper function for the mandelbrot kernel so that it compiles after inputting the above
    args

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
    pixel_to_point = pixel_to_point_wrapper(width, height, left, right, top, bottom)

    @cuda.jit
    def mandelbrot(data, offset_x, offset_y):
        id_x, id_y = identify_ids(offset_x, offset_y)
        c = transform(pixel_to_point(id_x, id_y))
        zn = z0

        escaped = False

        # cycle detection algorithm initial values
        check_step = 1
        epsilon = (right - left) / 1000000000  # scales as you zoom in
        zn_cycle = c

        for i in range(max_iter):
            zn = fn(zn, c)

            # finite iteration algorithm
            if zn.real ** 2 + zn.imag ** 2 > threshold ** 2:
                escaped = True
                # the smoothing factor
                nu = math.log(math.log(abs(zn)) / math.log(2.0)) / math.log(2.0)
                data[id_x, id_y] = i - nu
                break

            # cycle detection algorithm
            if i > check_step:
                if abs(zn - zn_cycle) < epsilon:
                    break
                if i == check_step * 2:
                    check_step *= 2
                    zn_cycle = zn

        if not escaped:
            data[id_x, id_y] = max_iter

    return mandelbrot


def julia_factory(width: int, height: int, left: float, right: float, top: float,
                  bottom: float, max_iter: int, threshold: float, z0: complex, fn,
                  transform, inv_transform, orbit_id):
    """A wrapper function for the julia kernel so that it compiles after inputting the above args

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
    pixel_to_point = pixel_to_point_wrapper(width, height, left, right, top, bottom)

    @cuda.jit
    def julia(data, offset_x, offset_y):
        id_x, id_y = identify_ids(offset_x, offset_y)

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


def julia_buddha_factory(width: int, height: int, left: float, right: float, top: float,
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
    pixel_to_point = pixel_to_point_wrapper(width, height, left, right, top, bottom)
    point_to_pixel = point_to_pixel_wrapper(width, height, left, right, top, bottom)

    @cuda.jit
    def julia_buddha(data, offset_x, offset_y):

        id_x, id_y = identify_ids(offset_x, offset_y)
        zn = transform(pixel_to_point(id_x, id_y))
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
    """A wrapper function for the orbit kernel so that it compiles after inputting the above args

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
    pixel_to_point = pixel_to_point_wrapper(width, height, left, right, top, bottom)

    @cuda.jit
    def orbits(data, offset_x, offset_y):

        id_x, id_y = identify_ids(offset_x, offset_y)
        c = transform(pixel_to_point(id_x, id_y))
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
    "julia_buddha": julia_buddha_factory,
    "orbit": orbit_factory}
