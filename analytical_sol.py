import matplotlib.pyplot as plt
import numpy as np

V0 = 10
R = 2
a = 8
h = 0.1
accuracy = 0.1 / 100
N = 7


def get_d(n):
    if n == 0:
        return 0
    else:
        return R ** 2 / (a - get_d(n - 1))


def get_q(n):
    if n == 0:
        return V0 * R
    else:
        return get_q(n - 1) * R / (a - get_d(n - 1))


def get_V(n, z):
    if abs(z - a / 2) < R:
        return V0
    else:
        return get_q(n) / abs(z + get_d(n) - a / 2) - get_q(n) / abs(
            z - get_d(n) + a / 2
        )


def calc_Vtot(n, z):
    voltage = 0
    if abs(z - a / 2) < R:
        return V0

    for i in range(1, n + 1):
        voltage += get_V(i, z)
    return voltage


def calc_iter():
    i = 1
    iters = []
    while True:
        iters.append(calc_Vtot(i, R))
        if i >= 2:
            err = abs(iters[i - 1] - iters[i - 2]) / abs(iters[i - 2])
            if err < accuracy:
                return i
        i += 1


def calc_V_r0():
    potential_vals = []
    domain = np.arange(0, 14 + h, h)
    for z in domain:
        potential_vals.append(calc_Vtot(N, z))

    plot_r(domain, potential_vals)


def plot_r(z_vals, potential_vals):
    plt.plot(z_vals, potential_vals)
    plt.xlabel("z")
    plt.ylabel("Potential")
    plt.show()


if __name__ == "__main__":
    print(calc_iter())
    calc_V_r0()
