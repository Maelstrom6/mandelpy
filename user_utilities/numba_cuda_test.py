from numba import cuda
import numpy as np


@cuda.jit
def double_data(data):
    id_x = cuda.threadIdx.x
    id_y = cuda.blockIdx.x
    data[id_x, id_y] = 1


shape = (5, 5)
output = np.zeros(shape, dtype=np.int)
double_data[shape[0], shape[1]](output)
worked = "Yes." if (output == np.ones(shape, dtype=np.int)).all() else "No."
print("Did it work?", worked)

print(cuda.detect())
