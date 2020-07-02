import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st
from scipy.optimize import minimize


def f(n: int, poly: list = [0, 1]):
    if n == 1:
        return poly
    else:

        def c(m):
            tot = 0
            for i in range(max(0, m - len(poly) + 1), min(m + 1, len(poly))):
                tot += poly[i] * poly[m - i]
            return tot

        new_len = 2 * len(poly) - 1
        new_poly = [0] * new_len
        for i in range(new_len):
            new_poly[i] = c(i)  # square
        new_poly[1] = 1  # add c

        return f(n - 1, new_poly)


# for n in range(1, 5):
#     print(f(n))
epsilon = 0.000001

observed = np.array(f(6))
s = np.sum(observed)
observed = observed / s
observed = observed[1:]
x = np.linspace(1, len(observed), len(observed), dtype=np.int)
print(x)

def get_quantiles(z):
    alpha, scale = z
    rv = st.gamma(alpha, loc=0., scale=scale)
    # rv = st.norm(loc=alpha, scale=scale)
    # rv = st.nbinom(n=alpha, p=scale)
    quantiles = rv.pdf(x)
    return quantiles


def mse(z):
    quantiles = get_quantiles(z)
    return np.mean((quantiles[::-1] - observed) ** 2)


def plot(z):
    quantiles = get_quantiles(z)
    plt.plot(observed, c="red")
    plt.plot(quantiles[::-1], c="blue")
    plt.show()

result = minimize(mse, np.array([2, 2]), bounds=[(epsilon, 5000), (epsilon, 5000)])
# result = minimize(mse, np.array([500, 0.97]), bounds=[(epsilon, 5000), (epsilon, 1-epsilon)])
print(result)
plot(result.x)
# plot([550, 0.97588774])

