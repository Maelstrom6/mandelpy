import numpy as np


def color(array: np.ndarray, tipe: str, scheme_id: int, max_iter: int):
    if tipe == "mand":
        array = color_mand(array, scheme_id, max_iter)
    elif (tipe == "buddha") or (tipe == "antibuddha"):
        array = color_buddha(array, scheme_id)
    return array


def color_mand(array: np.ndarray, scheme_id: int, max_iter: int):

    new_array = np.copy(array)

    if scheme_id == 0:
        new_array[:, :, 0] = np.sin(new_array[:, :, 0] / 24) * 100 + 150
        new_array[:, :, 1] = np.sin(new_array[:, :, 1] / 12) * 100 + 150
        new_array[:, :, 2] = np.cos(new_array[:, :, 2] / 24) * 100 + 150

    else:
        new_array[:, :, 0] = np.sin(new_array[:, :, 0] / 24) * 100 + 150
        new_array[:, :, 1] = np.sin(new_array[:, :, 1] / 12) * 100 + 150
        new_array[:, :, 2] = np.cos(new_array[:, :, 2] / 24) * 100 + 150

    np.putmask(new_array, array == max_iter, 0)
    return new_array


def color_buddha(array: np.ndarray, scheme_id: int):
    m = array.max((0, 1), initial=1)  # Get the max of the rgb

    if scheme_id == 0:
        array = 255 * np.sqrt(array / m)  # scale the colors

    elif scheme_id == 1:
        array = 255 * np.sqrt(array / m)
        tmp = array[:, :, 0]
        array[:, :, 0] = array[:, :, 1]
        array[:, :, 1] = array[:, :, 2]
        array[:, :, 2] = tmp

    elif scheme_id == 2:
        array = 255 * np.sqrt(array / m)
        tmp = array[:, :, 0]
        array[:, :, 0] = array[:, :, 2]
        array[:, :, 2] = array[:, :, 1]
        array[:, :, 1] = tmp

    elif scheme_id == 3:
        array = 255 * np.log(array+1)/np.log(m+1)

    else:
        array = 255 * np.sqrt(array / m)
    return array
