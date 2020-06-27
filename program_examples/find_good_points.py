from scipy.optimize import minimize, newton
import numpy as np
from matplotlib import pyplot as plt
import random
import math

n = 7
epsilon = 0.00011


def g(c):
    z = 0
    c = complex(c[0], c[1])

    f = lambda zn, c: zn ** 2 + c
    for iteration in range(n):
        z = f(z, c)

        if abs(z)>2:
            return abs(z)**(n-iteration)

    return abs(z)

for n in range(1, 10):
    zeroes = [complex(0, 0)]
    step = 0.2
    for x in np.arange(-2, 2 + step, step):
        for y in np.arange(-2, 2 + step, step):
            if x**2 + y**2 >= 4:
                continue
            initial = np.array([x, y])
            try:
                zero = minimize(g, initial).x
                zero = complex(zero[0], zero[1])
                distance = min([abs(zero - z) for z in zeroes])
                if distance > epsilon:
                    zeroes.append(zero)
            except RuntimeError:
                pass

    reals = [z.real for z in zeroes]
    imags = [z.imag for z in zeroes]
    plt.scatter(reals, imags)
    plt.xlim((-2, 1.5))
    plt.ylim((-1.75, 1.75))
    plt.savefig(f"{n}.png")
