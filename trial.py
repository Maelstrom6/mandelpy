from cmath import *
import numpy as np

t = lambda z: tan(asin(z)) ** 2
it = lambda z: sin(atan(sqrt(z)))

for x in np.arange(-2, 2):
    for y in np.arange(-2, 2):
        print(it(t(complex(x, y))))
        print(it(t(complex(x, -y))))
        print()