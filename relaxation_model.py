import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self, size, max_diff=0.00001, start_value=1, max_iter=100000):
        self._current_potential = np.ones((size, size)) * start_value
        self._delta_x = 1 / size
        self._impose_boundary(self._current_potential)
        self._max_diff = max_diff
        self._max_iter = max_iter

    @property
    def current_potential(self):
        return self._current_potential

    @property
    def max_iter(self):
        return self._max_iter

    @property
    def max_diff(self):
        return self._max_diff

    def solve(self):
        max_diff = np.inf
        iter_amount = 0
        while max_diff > self.max_diff and iter_amount < self.max_iter:
            max_diff = self._make_step()
            iter_amount += 1

        return self.current_potential, self._get_velocity(self.current_potential)

    def _make_step(self):
        new_potential = np.copy(self._current_potential)
        (n, m) = self.current_potential.shape
        new_potential[1 : n - 1, 1 : m - 1] = (
            self.current_potential[0 : n - 2, 1 : m - 1]
            + self.current_potential[2:n, 1 : m - 1]
            + self.current_potential[1 : n - 1, 0 : m - 2]
            + self.current_potential[1 : n - 1, 2:m]
        ) / 4
        self._impose_boundary(new_potential)
        max_diff = np.max(np.abs(new_potential - self.current_potential))
        self._current_potential = new_potential
        return max_diff

    def _impose_boundary(self, potential):
        (n, m) = potential.shape
        # all sides
        potential[:, 0:1] = potential[:, 1:2]
        potential[n - 1 : n, 0:1] = potential[n - 1 : n, 1:2]
        potential[0:1, :] = potential[0:1, :]
        potential[0:1, m - 1 : m] = potential[0:1, m - 1 : m]
        # left corner
        potential[n // 4 : 3 * n // 4, 0:1] = (
            potential[n // 4 : 3 * n // 4, 1:2] - self._delta_x * 100
        )
        # right corner
        potential[:, n - 1 : n] = potential[:, n - 2 : n - 1] + self._delta_x * 100 / 2

    def _get_velocity(self, potential):
        (n, m) = potential.shape
        velocity = np.zeros((n, m))
        for x in range(n):
            for y in range(m):
                if x == 0:
                    velocity[y, x] = (
                        potential[y, x + 1] - potential[y, x]
                    ) / self._delta_x
                elif x == n - 1:
                    velocity[y, x] = (
                        potential[y, x] - potential[y, x-1]
                    ) / self._delta_x
                else:
                    velocity[y, x] = (potential[y, x + 1] - potential[y, x-1]) / (
                        2 * self._delta_x
                    )
        return velocity


def main():
    model = Model(100)
    potential, velocity = model.solve()
    plt.imshow(velocity, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
