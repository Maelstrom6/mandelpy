"""
The main class for validating user input from the GUI.
"""

from typing import Callable, List
import ast
from cmath import *  # this is needed for validate_function
from numba import cuda

integer_error_message = "This value must be an integer."
positive_error_message = "This value must be positive."
float_error_message = "This value must be a float."
function_error_message = "Must be a function of "
file_error_message = "This must end with "
complex_error_message = "This must be a tuple like 1, 0."
block_error_message = "Should be a tuple like 500, 500."


@cuda.jit(device=True)
def power(z, n):
    """Finds z^n using CUDA-supported functions"""
    return exp(n * log(z))


def validate_int(x: str, positive: bool = False) -> int:
    try:
        y = int(x)
    except ValueError:
        raise ValueError(integer_error_message)
    if (y <= 0) and positive:
        raise ValueError(positive_error_message)
    return y


def validate_float(x: str, positive: bool = False) -> float:
    try:
        y = float(x)
    except ValueError:
        raise ValueError(float_error_message)
    if (y <= 0) and positive:
        raise ValueError(positive_error_message)
    return y


def validate_function(f: str, valid_variables: List[str]) -> Callable:
    # choose args that are unlikely to have an asymptote (so that it will be defined)
    sample_args = [0.999] * len(valid_variables)
    to_eval = fr"lambda {', '.join(valid_variables)}: {f}"
    y = eval(to_eval)

    if "power(" not in f:  # power function cannot be called. Ignore evaluation
        try:
            # evaluate the function with sample args so that it can throw if needed
            y(*sample_args)
        except NameError:
            raise ValueError(function_error_message + ", ".join(valid_variables) + ".")
        except ZeroDivisionError:
            pass  # we chose args that gave an asymptote. Not a problem

    return y


def validate_file_name(file_name: str, file_formats: List[str] = None) -> str:
    # apply default argument
    if file_formats is None:
        file_formats = ["png"]

    # check if file extension is correct
    for file_format in file_formats:
        if not file_name.lower().endswith(file_format):
            raise ValueError(file_error_message + " or ".join(file_formats))

    return file_name


def validate_complex(x: str) -> complex:
    try:
        y = eval(rf"complex({x})")
    except:
        raise ValueError(complex_error_message)
    return y


def validate_block_size(x: str) -> tuple:
    try:
        y = ast.literal_eval(fr"({x})")
        if len(y) != 2:
            raise ValueError
        for i in range(2):
            if (not isinstance(y[i], int)) or y[i] <= 0:
                raise ValueError
    except:
        raise ValueError(block_error_message)
    return y
