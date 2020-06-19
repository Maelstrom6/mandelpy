from numba import cuda
import numpy as np
z = complex(0, 0)


@cuda.jit
def double_data(data):
    id_x = cuda.threadIdx.x
    id_y = cuda.blockIdx.x
    data[id_x, id_y] = 1.2


print(type(double_data))
